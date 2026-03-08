# Data manipulation
import pandas as pd
import numpy as np

# String cleaning
import re

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score
from sklearn.preprocessing import StandardScaler

# Gradient Boosting Ranker
from lightgbm import LGBMRanker

# visualization
import matplotlib.pyplot as plt
import seaborn as sns

# # Read datasets
# recipes = pd.read_csv("recipes.csv")
# produce_prices = pd.read_csv("produce_prices.csv")
# produce_seasonality = pd.read_csv("produce_seasonality.csv")

# ----------------------------
# 1) Read the CSV files
# ----------------------------
recipes = pd.read_csv("recipes.csv")
produce_prices = pd.read_csv("produce_prices.csv")
produce_seasonality = pd.read_csv("produce_seasonality.csv")

print("recipes:", recipes.shape)
print("produce_prices:", produce_prices.shape)
print("produce_seasonality:", produce_seasonality.shape)

# ----------------------------
# 2) Clean + split ingredients
#    Assumes ingredients are comma-separated or list-like text
# ----------------------------
def parse_ingredients(text):
    if pd.isna(text):
        return []
    s = str(text).lower().strip()

    # Remove common list wrappers and quotes
    s = re.sub(r"[\[\]\(\)\"']", "", s)

    # Split on commas or semicolons
    parts = re.split(r",|;", s)

    cleaned = []
    for p in parts:
        item = p.strip()

        # Drop empty items
        if not item:
            continue

        # Basic cleanup: remove leading quantities (e.g., "2 cups", "1/2", "3 tbsp")
        item = re.sub(r"^\s*\d+(\.\d+)?\s*", "", item)   # leading numbers
        item = re.sub(r"^\s*\d+/\d+\s*", "", item)       # leading fractions
        item = item.strip()

        if item:
            cleaned.append(item)

    return cleaned

# Create a list column
recipes["ingredient_list"] = recipes["ingredients"].apply(parse_ingredients)

# ----------------------------
# 3) Explode to one row per ingredient
# ----------------------------
# If your file has a recipe id column, use it. If not, create one.
if "recipe_id" not in recipes.columns:
    recipes = recipes.reset_index().rename(columns={"index": "recipe_id"})

recipe_ingredients = recipes[["recipe_id", "ingredient_list"]].explode("ingredient_list")
recipe_ingredients = recipe_ingredients.rename(columns={"ingredient_list": "produce"})

# Final cleanup
recipe_ingredients["produce"] = recipe_ingredients["produce"].astype(str).str.strip()
recipe_ingredients = recipe_ingredients[recipe_ingredients["produce"] != ""]

print("recipe_ingredients (exploded):", recipe_ingredients.shape)
recipe_ingredients.head(10)

# ----------------------------
# 4) Pick a query month (1–12)
#    (Later we’ll train across all months; for now pick one)
# ----------------------------
QUERY_MONTH = 3  # March example

# Add month to each (recipe, produce) row
recipe_ingredients["month"] = QUERY_MONTH

# ----------------------------
# 5) Clean keys + dedupe external tables
# ----------------------------
produce_prices["produce"] = produce_prices["produce"].astype(str).str.lower().str.strip()
produce_seasonality["produce"] = produce_seasonality["produce"].astype(str).str.lower().str.strip()

# Ensure month is int
produce_prices["month"] = produce_prices["month"].astype(int)
produce_seasonality["month"] = produce_seasonality["month"].astype(int)

# Dedupe in case there are duplicates
produce_prices = produce_prices.drop_duplicates(subset=["produce", "month"])
produce_seasonality = produce_seasonality.drop_duplicates(subset=["produce", "month"])

# ----------------------------
# 6) Merge: (recipe, produce, month) -> in_season + price
# ----------------------------
df = recipe_ingredients.merge(
    produce_seasonality.assign(in_season=1),   # seasonality file is just presence = in season
    on=["produce", "month"],
    how="left"
)

df["in_season"] = df["in_season"].fillna(0).astype(int)

df = df.merge(
    produce_prices[["produce", "month", "price_per_unit", "unit"]],
    on=["produce", "month"],
    how="left"
)

# If a price is missing, we’ll mark it and optionally fill later
df["missing_price"] = df["price_per_unit"].isna().astype(int)

# Simple fill for missing prices (keeps pipeline moving):
# fill missing with the median price across all produce that month
median_price = df["price_per_unit"].median()
df["price_per_unit"] = df["price_per_unit"].fillna(median_price)

print("Merged ingredient-level rows:", df.shape)
df.head(10)


# ----------------------------
# 7) Compute recipe-level features for ranking
# ----------------------------
recipe_features = df.groupby("recipe_id").agg(
    total_cost=("price_per_unit", "sum"),
    seasonality_rate=("in_season", "mean"),      # fraction of ingredients in season
    missing_price_count=("missing_price", "sum"),
    ingredient_count=("produce", "count")
).reset_index()

# Optional: a combined score baseline (non-ML) for comparison
# Higher seasonality_rate is better; lower cost is better
recipe_features["baseline_score"] = (
    recipe_features["seasonality_rate"] * 1.0
    - 0.05 * recipe_features["total_cost"]
    - 0.10 * recipe_features["missing_price_count"]
)

recipe_features.sort_values("baseline_score", ascending=False).head(10)

# ----------------------------
# 8) Expand recipe ingredients across ALL months (1–12)
# ----------------------------
months = pd.DataFrame({"month": list(range(1, 13))})

# Cross join: every (recipe, produce) gets duplicated for every month
recipe_ingredients_all = recipe_ingredients.drop(columns=["month"], errors="ignore").merge(months, how="cross")

# Merge with seasonality + prices
df_all = recipe_ingredients_all.merge(
    produce_seasonality.assign(in_season=1),
    on=["produce", "month"],
    how="left"
)
df_all["in_season"] = df_all["in_season"].fillna(0).astype(int)

df_all = df_all.merge(
    produce_prices[["produce", "month", "price_per_unit"]],
    on=["produce", "month"],
    how="left"
)

df_all["missing_price"] = df_all["price_per_unit"].isna().astype(int)

# Fill missing prices using month-specific medians (more stable than global)
df_all["price_per_unit"] = df_all.groupby("month")["price_per_unit"].transform(
    lambda s: s.fillna(s.median())
)

print("Ingredient-level rows across all months:", df_all.shape)
df_all.head(10)

# ----------------------------
# 9) Recipe-level features per (recipe_id, month)
# ----------------------------
features = df_all.groupby(["recipe_id", "month"]).agg(
    total_cost=("price_per_unit", "sum"),
    seasonality_rate=("in_season", "mean"),
    missing_price_count=("missing_price", "sum"),
    ingredient_count=("produce", "count"),
).reset_index()

# A helpful engineered feature:
features["cost_per_ingredient"] = features["total_cost"] / features["ingredient_count"]

features.head(10)

# ----------------------------
# 10) Create ranking labels (0,1,2) per month
# ----------------------------
def label_month(group):
    cost_q30 = group["total_cost"].quantile(0.30)
    cost_q70 = group["total_cost"].quantile(0.70)
    seas_q30 = group["seasonality_rate"].quantile(0.30)
    seas_q70 = group["seasonality_rate"].quantile(0.70)

    labels = np.ones(len(group), dtype=int)

    best = (group["total_cost"] <= cost_q30) & (group["seasonality_rate"] >= seas_q70)
    worst = (group["total_cost"] >= cost_q70) & (group["seasonality_rate"] <= seas_q30)

    labels[best.values] = 2
    labels[worst.values] = 0
    return labels

features["label"] = features.groupby("month", group_keys=False).apply(label_month)

# Make sure label is numeric (int)
features["label"] = pd.to_numeric(features["label"], errors="coerce").fillna(1).astype(int)

features[["recipe_id", "month", "total_cost", "seasonality_rate", "label"]].head(10)

from lightgbm import LGBMRanker

# ----------------------------
# 11) Train/test split by month
# ----------------------------
train_months = [1,2,3,4,5,6,7,8,9]    # train on 9 months
test_months  = [10,11,12]            # test on last 3 months

train_df = features[features["month"].isin(train_months)].copy()
test_df  = features[features["month"].isin(test_months)].copy()

feature_cols = ["total_cost", "seasonality_rate", "missing_price_count", "ingredient_count", "cost_per_ingredient"]

X_train = train_df[feature_cols]
y_train = train_df["label"]

X_test = test_df[feature_cols]
y_test = test_df["label"]

# Group sizes: number of recipes per month
group_train = train_df.groupby("month").size().tolist()
group_test = test_df.groupby("month").size().tolist()

# ----------------------------
# 12) Fit ranker
# ----------------------------
ranker = LGBMRanker(
    objective="lambdarank",
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42
)

ranker.fit(
    X_train, y_train,
    group=group_train
)

print("Model trained.")

############################
# 13) Evaluate on test set

print("Label distribution overall:")
print(features["label"].value_counts(normalize=True))

print("\nLabel distribution by month:")
print(features.groupby("month")["label"].value_counts().unstack(fill_value=0))

print("\nFeature variance check:")
print(features[feature_cols].describe().T[["mean", "std", "min", "max"]])

###################################
# What % of ingredient rows matched seasonality?

print("Matched seasonality %:",
      df_all["in_season"].mean())

# How many unique ingredients in recipes?
print("Unique ingredients in recipes:",
      recipe_ingredients["produce"].nunique())

# How many unique produce in seasonality file?
print("Unique produce in seasonality:",
      produce_seasonality["produce"].nunique())


###################################
# Produce detection layer 

ingredient_counts = recipe_ingredients["produce"].value_counts()
ingredient_counts.head(50)

produce_keywords = [
    "onion", "garlic", "tomato", "pepper", "spinach",
    "lettuce", "carrot", "zucchini", "cucumber",
    "apple", "lemon", "lime", "orange", "berry",
    "basil", "cilantro", "parsley", "celery",
    "cabbage", "broccoli", "cauliflower",
    "mushroom", "ginger", "potato"
]

df_all["is_produce"] = df_all["produce"].apply(
    lambda x: any(k in x for k in produce_keywords)
).astype(int)

ingredient_counts.head(50)