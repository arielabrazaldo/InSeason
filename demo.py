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