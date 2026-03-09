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

# seasonality joins are failing because ingredient strings in receipes dont match produce tables
def normalize_produce(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"[^a-z\s]", " ", s)            # remove punctuation/numbers
    s = re.sub(r"\s+", " ", s).strip()         # collapse spaces
    # remove common cooking descriptors
    drop = ["fresh", "frozen", "chopped", "diced", "minced", "sliced", "ground", "organic"]
    for w in drop:
        s = re.sub(rf"\b{w}\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

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
recipe_ingredients["produce"] = recipe_ingredients["produce"].apply(normalize_produce)
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

produce_prices["produce"] = produce_prices["produce"].apply(normalize_produce)
produce_seasonality["produce"] = produce_seasonality["produce"].apply(normalize_produce)

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

# Debug 1
print("\nDEBUG: Ingredient-level in-season match rate")
print("in_season mean:", df_all["in_season"].mean())

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

# Debug 3
print("\nDEBUG: Train label distribution")
print(train_df["label"].value_counts())

print("\nDEBUG: Test label distribution")
print(test_df["label"].value_counts())

feature_cols = ["total_cost", "seasonality_rate", "missing_price_count", "ingredient_count", "cost_per_ingredient"]

X_train = train_df[feature_cols]
y_train = train_df["label"]

X_test = test_df[feature_cols]
y_test = test_df["label"]

# Group sizes: number of recipes per month
group_train = train_df.groupby("month").size().tolist()
group_test = test_df.groupby("month").size().tolist()

# commented out first training section for testing
'''
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
'''

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

# print("Matched seasonality %:",
#       df_all["in_season"].mean())

# # How many unique ingredients in recipes?
# print("Unique ingredients in recipes:",
#       recipe_ingredients["produce"].nunique())

# # How many unique produce in seasonality file?
# print("Unique produce in seasonality:",
#       produce_seasonality["produce"].nunique())

###################################
# Produce detection layer 

ingredient_counts = recipe_ingredients["produce"].value_counts()
ingredient_counts.head(50)

produce_keywords = [
    # Alliums
    "onion", "garlic", "shallot", "scallion", "leek",
    # Nightshades
    "tomato", "pepper", "chili", "jalapeno", "bell pepper",
    # Leafy greens
    "spinach", "lettuce", "arugula", "kale", "chard", "collard",
    # Root vegetables
    "carrot", "beet", "radish", "turnip", "parsnip",
    "potato", "sweet potato", "yam",
    # Squash
    "zucchini", "squash", "pumpkin",
    # Cruciferous
    "broccoli", "cauliflower", "cabbage", "brussels sprout",
    # Herbs
    "basil", "cilantro", "parsley", "thyme", "rosemary",
    "oregano", "dill", "mint",
    # Fruits
    "apple", "pear", "peach", "plum", "nectarine",
    "orange", "lemon", "lime", "grape",
    "berry", "strawberry", "blueberry", "raspberry",
    "blackberry", "mango", "pineapple",
    "watermelon", "cantaloupe", "melon",
    # Other common produce
    "cucumber", "celery", "mushroom", "ginger",
    "avocado", "corn", "eggplant"
]

df_all["is_produce"] = df_all["produce"].apply(
    lambda x: any(k in x for k in produce_keywords)
).astype(int)

# -------------------------------------------------
# Step 2: Recompute features correctly
# -------------------------------------------------

features = df_all.groupby(["recipe_id", "month"]).agg(
    total_cost=("price_per_unit", "sum"),
    produce_count=("is_produce", "sum"),
    in_season_produce=("in_season", "sum"),
    missing_price_count=("missing_price", "sum"),
    ingredient_count=("produce", "count"),
).reset_index()

features["seasonality_rate"] = np.where(
    features["produce_count"] > 0,
    features["in_season_produce"] / features["produce_count"],
    0
)

# debug 2
print("\nDEBUG: Recipe-level seasonality stats")
print(features["seasonality_rate"].describe())

features["cost_per_ingredient"] = features["total_cost"] / features["ingredient_count"]

# print(features["seasonality_rate"].describe())

# Check how many recipes have at least 1 produce ingredient

# print("Percent of recipes with at least 1 produce ingredient:",
#       (features["produce_count"] > 0).mean())

# print("Average produce_count per recipe:",
#       features["produce_count"].mean())


# Normalize cost within month
features["cost_z"] = features.groupby("month")["total_cost"].transform(
    lambda s: (s - s.mean()) / (s.std() + 1e-9)
)

# Higher is better: more in-season, lower cost
# **************************************************
# seasonlity_rate is mostly 0 due to mapping issues
# train using cost only so labels don't collapse
# changing this:
# features["target_score"] = features["seasonality_rate"] - 0.4 * features["cost_z"]
# makes labels depend on cost only, which should immediately produce 0/1/2 lables and stop LightGBM warnings
features["target_score"] = -features["cost_z"]

# Balanced 3-class labels per month using ranks (robust even with ties)
def make_tertiles_by_rank(group):
    r = group["target_score"].rank(method="first")  # breaks ties deterministically
    q1 = r.quantile(0.33)
    q2 = r.quantile(0.66)
    return np.select([r <= q1, r <= q2], [0, 1], default=2).astype(int)

features["label"] = features.groupby("month", group_keys=False).apply(make_tertiles_by_rank)
features["label"] = features["label"].astype(int)

print("\nDEBUG: Overall label distribution")
print(features["label"].value_counts())

print("Label distribution:")
print(features["label"].value_counts(normalize=True))

# --- confirms whether seasonlity is bascially always 0 and whether labels are balanced ---
print("\nLabel counts (raw):")
print(features["label"].value_counts())

print("\nSeasonality stats:")
print(features["seasonality_rate"].describe())

print("\nIn-season match rate at ingredient-row level:")
print("in_season mean:", df_all["in_season"].mean())

# Training LightGBM training with the new features and labels

# Train/test split by month
train_months = [1,2,3,4,5,6,7,8,9]
test_months  = [10,11,12]

train_df = features[features["month"].isin(train_months)].copy()
test_df  = features[features["month"].isin(test_months)].copy()

# --- verify that labels aren't collapsing right before training ---
print("\nTrain label distribution:")
print(train_df["label"].value_counts())

print("\nTest label distribution:")
print(test_df["label"].value_counts())

feature_cols = [
    "total_cost",
    "seasonality_rate",
    "missing_price_count",
    "ingredient_count",
    "cost_per_ingredient"
]

X_train = train_df[feature_cols]
y_train = train_df["label"]

X_test = test_df[feature_cols]
y_test = test_df["label"]

group_train = train_df.groupby("month").size().tolist()

# ---changing min_data_in_leaf to 20 more stable to reduce "no positive gain" warnings when features are weak ---
ranker = LGBMRanker(
    objective="lambdarank",
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=63,
    min_data_in_leaf=20,
    random_state=42
)

ranker.fit(
    X_train,
    y_train,
    group=group_train
)

print("Model trained successfully.")


# Measuring quality 

# Predict on test
test_df["pred"] = ranker.predict(X_test)

def ndcg_at_k(df_month, k=10):
    y_true = df_month["label"].values.reshape(1, -1)
    y_score = df_month["pred"].values.reshape(1, -1)
    return ndcg_score(y_true, y_score, k=k)

results = []

for m, g in test_df.groupby("month"):
    score = ndcg_at_k(g, k=10)
    results.append((m, score))

ndcg_results = pd.DataFrame(results, columns=["month", "ndcg@10"])
print(ndcg_results)
print("\nAverage NDCG@10:", ndcg_results["ndcg@10"].mean())

# baseliine ranking ################

# Baseline score (no ML)
test_df["baseline_score"] = (
    test_df["seasonality_rate"] - 0.4 * test_df["cost_z"]
)

def ndcg_baseline(df_month, k=10):
    y_true = df_month["label"].values.reshape(1, -1)
    y_score = df_month["baseline_score"].values.reshape(1, -1)
    return ndcg_score(y_true, y_score, k=k)

baseline_results = []

for m, g in test_df.groupby("month"):
    score = ndcg_baseline(g, k=10)
    baseline_results.append((m, score))

baseline_df = pd.DataFrame(baseline_results, columns=["month", "baseline_ndcg@10"])
print(baseline_df)
print("\nBaseline Average NDCG@10:", baseline_df["baseline_ndcg@10"].mean())

# inspect feature importance

import matplotlib.pyplot as plt

importances = ranker.feature_importances_
feature_importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": importances
}).sort_values("importance", ascending=False)

print(feature_importance_df)

# feature_importance_df.plot.bar(x="feature", y="importance")
# plt.title("Feature Importance")
# plt.show()

####  inspected top-ranked recipes

month_to_view = 12

month_df = test_df[test_df["month"] == month_to_view].copy()
month_df = month_df.sort_values("pred", ascending=False)

top10 = month_df.head(10)[
    ["recipe_id", "total_cost", "seasonality_rate", "produce_count"]
]

print(top10)

# additional label debugging outputs
print(features["label"].value_counts())
print(train_df["label"].value_counts())