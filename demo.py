# Data manipulation
import pandas as pd
import numpy as np
import re
from sklearn.metrics import ndcg_score
from lightgbm import LGBMRanker
import matplotlib.pyplot as plt
from scipy import stats

# ----------------------------
# 1) Read CSV files
# ----------------------------
recipes = pd.read_csv("recipes.csv")
produce_prices = pd.read_csv("produce_prices.csv")
produce_seasonality = pd.read_csv("produce_seasonality.csv")

print("recipes:", recipes.shape)
print("produce_prices:", produce_prices.shape)
print("produce_seasonality:", produce_seasonality.shape)

# ----------------------------
# 2) Parse ingredients
# ----------------------------
def parse_ingredients(text):
    if pd.isna(text):
        return []
    s = str(text).lower().strip()
    s = re.sub(r"[\[\]\(\)\"']", "", s)
    parts = re.split(r",|;", s)
    
    cleaned = []
    for p in parts:
        item = p.strip()
        if not item:
            continue
        item = re.sub(r"^\s*\d+(\.\d+)?\s*", "", item)
        item = re.sub(r"^\s*\d+/\d+\s*", "", item)
        item = item.strip()
        if item:
            cleaned.append(item)
    return cleaned

recipes["ingredient_list"] = recipes["ingredients"].apply(parse_ingredients)

# ----------------------------
# 3) Create recipe_id and explode ingredients
# ----------------------------
if "recipe_id" not in recipes.columns:
    recipes = recipes.reset_index().rename(columns={"index": "recipe_id"})

recipe_ingredients = recipes[["recipe_id", "ingredient_list"]].explode("ingredient_list")
recipe_ingredients = recipe_ingredients.rename(columns={"ingredient_list": "produce"})
recipe_ingredients["produce"] = recipe_ingredients["produce"].astype(str).str.strip()
recipe_ingredients = recipe_ingredients[recipe_ingredients["produce"] != ""]

print("recipe_ingredients (exploded):", recipe_ingredients.shape)

# ----------------------------
# 4) Clean external data
# ----------------------------
produce_prices["produce"] = produce_prices["produce"].astype(str).str.lower().str.strip()
produce_seasonality["produce"] = produce_seasonality["produce"].astype(str).str.lower().str.strip()
produce_prices["month"] = produce_prices["month"].astype(int)
produce_seasonality["month"] = produce_seasonality["month"].astype(int)
produce_prices = produce_prices.drop_duplicates(subset=["produce", "month"])
produce_seasonality = produce_seasonality.drop_duplicates(subset=["produce", "month"])

# ----------------------------
# 5) Expand across all months
# ----------------------------
months = pd.DataFrame({"month": list(range(1, 13))})
recipe_ingredients_all = recipe_ingredients.merge(months, how="cross")

# ----------------------------
# 6) Produce detection BEFORE merging
# ----------------------------
produce_keywords = [
    "onion", "garlic", "shallot", "scallion", "leek",
    "tomato", "pepper", "chili", "jalapeno", "bell pepper",
    "spinach", "lettuce", "arugula", "kale", "chard", "collard",
    "carrot", "beet", "radish", "turnip", "parsnip",
    "potato", "sweet potato", "yam",
    "zucchini", "squash", "pumpkin",
    "broccoli", "cauliflower", "cabbage", "brussels sprout",
    "basil", "cilantro", "parsley", "thyme", "rosemary",
    "oregano", "dill", "mint",
    "apple", "pear", "peach", "plum", "nectarine",
    "orange", "lemon", "lime", "grape",
    "berry", "strawberry", "blueberry", "raspberry",
    "blackberry", "mango", "pineapple",
    "watermelon", "cantaloupe", "melon",
    "cucumber", "celery", "mushroom", "ginger",
    "avocado", "corn", "eggplant"
]

recipe_ingredients_all["is_produce"] = recipe_ingredients_all["produce"].apply(
    lambda x: any(k in x for k in produce_keywords)
).astype(int)

# ----------------------------
# 7) Merge with seasonality and prices
# ----------------------------
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
df_all["price_per_unit"] = df_all.groupby("month")["price_per_unit"].transform(
    lambda s: s.fillna(s.median())
)

print("Ingredient-level rows across all months:", df_all.shape)

# ----------------------------
# 8) Create recipe-level features (ONCE!)
# ----------------------------
features = df_all.groupby(["recipe_id", "month"]).agg(
    total_cost=("price_per_unit", "sum"),
    produce_count=("is_produce", "sum"),
    in_season_produce=("in_season", "sum"),
    missing_price_count=("missing_price", "sum"),
    ingredient_count=("produce", "count"),
).reset_index()

# Calculate seasonality rate (only for produce)
features["seasonality_rate"] = np.where(
    features["produce_count"] > 0,
    features["in_season_produce"] / features["produce_count"],
    0
)

features["cost_per_ingredient"] = features["total_cost"] / features["ingredient_count"]

# ----------------------------
# 9) Create REALISTIC relevance labels (not just linear combination)
# ----------------------------
features["cost_normalized"] = features.groupby("month")["total_cost"].transform(
    lambda s: (s - s.min()) / (s.max() - s.min() + 1e-9)
)

features["relevance_score"] = (
    0.6 * features["seasonality_rate"] +
    0.4 * (1 - features["cost_normalized"])
)

# Convert to 5-level integer labels for LambdaRank
features["relevance_label"] = features.groupby("month")["relevance_score"].transform(
    lambda s: pd.qcut(s, q=5, labels=[0, 1, 2, 3, 4], duplicates='drop')
).astype(int)

print("\nRelevance score distribution:")
print(features["relevance_score"].describe())
print("\nRelevance label distribution:")
print(features["relevance_label"].value_counts().sort_index())
# ----------------------------
# 10) Train/test split
# ----------------------------
train_months = [1,2,3,4,5,6,7,8,9]
test_months = [10,11,12]

train_df = features[features["month"].isin(train_months)].copy()
test_df = features[features["month"].isin(test_months)].copy()

feature_cols = [
    "total_cost",
    "seasonality_rate",
    "missing_price_count",
    "ingredient_count",
    "cost_per_ingredient"
]

X_train = train_df[feature_cols]
y_train = train_df["relevance_label"]  # ← Integer labels

X_test = test_df[feature_cols]
y_test = test_df["relevance_label"]

group_train = train_df.groupby("month").size().tolist()
group_test = test_df.groupby("month").size().tolist()

# ----------------------------
# 11) Train ML Ranker
# ----------------------------
print("\n" + "="*70)
print("Training ML Ranker...")
print("="*70)

ranker = LGBMRanker(
    objective="lambdarank",
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=63,
    min_data_in_leaf=5,
    random_state=42,
    verbosity=-1
)

ranker.fit(X_train, y_train, group=group_train)
print("✓ Model trained successfully")

# Predict
test_df["pred_ml"] = ranker.predict(X_test)

# ----------------------------
# 12) Evaluate ML Ranker (using continuous scores as ground truth)
# ----------------------------
def evaluate_ranker(df_test, score_col, label="ML Ranker"):
    results = []
    for m, g in df_test.groupby("month"):
        y_true = g["relevance_score"].values.reshape(1, -1)  # Use continuous score
        y_pred = g[score_col].values.reshape(1, -1)
        score = ndcg_score(y_true, y_pred, k=10)
        results.append((m, score))
    return pd.DataFrame(results, columns=["month", "ndcg@10"])

ml_ndcg = evaluate_ranker(test_df, "pred_ml", "ML Ranker")
print("\n=== ML Ranker Results ===")
print(ml_ndcg)
print(f"Average NDCG@10: {ml_ndcg['ndcg@10'].mean():.4f}")

# ----------------------------
# 13) Baseline 1: Cost-only
# ----------------------------
cost_only_results = []
for m, g in test_df.groupby("month"):
    y_true = g["relevance_score"].values.reshape(1, -1)
    y_score = (-g["total_cost"]).values.reshape(1, -1)
    score = ndcg_score(y_true, y_score, k=10)
    cost_only_results.append((m, score))

cost_only_ndcg = pd.DataFrame(cost_only_results, columns=["month", "ndcg@10"])
print("\n=== Cost-Only Ranking ===")
print(cost_only_ndcg)
print(f"Average NDCG@10: {cost_only_ndcg['ndcg@10'].mean():.4f}")

# ----------------------------
# 14) Baseline 2: Seasonality-only
# ----------------------------
seasonality_only_results = []
for m, g in test_df.groupby("month"):
    y_true = g["relevance_score"].values.reshape(1, -1)
    y_score = g["seasonality_rate"].values.reshape(1, -1)
    score = ndcg_score(y_true, y_score, k=10)
    seasonality_only_results.append((m, score))

seasonality_only_ndcg = pd.DataFrame(seasonality_only_results, columns=["month", "ndcg@10"])
print("\n=== Seasonality-Only Ranking ===")
print(seasonality_only_ndcg)
print(f"Average NDCG@10: {seasonality_only_ndcg['ndcg@10'].mean():.4f}")

# ----------------------------
# 15) Baseline 3: Simple weighted
# ----------------------------

# ----------------------------
# 16) Comparison table
# ----------------------------
comparison = pd.DataFrame({
    "Month": ml_ndcg["month"],
    "ML Ranker": ml_ndcg["ndcg@10"],
    "Cost Only": cost_only_ndcg["ndcg@10"],
    "Seasonality Only": seasonality_only_ndcg["ndcg@10"],
    #"Simple Weighted": simple_weighted_ndcg["ndcg@10"]
})

print("\n" + "="*70)
print("COMPARISON: NDCG@10 Across All Methods")
print("="*70)
print(comparison.to_string(index=False))

avg_comparison = pd.DataFrame({
    "Method": ["ML Ranker", "Cost Only", "Seasonality Only"],
    "Avg NDCG@10": [
        comparison["ML Ranker"].mean(),
        comparison["Cost Only"].mean(),
        comparison["Seasonality Only"].mean(),
    ]
}).sort_values("Avg NDCG@10", ascending=False)

print("\n" + "="*70)
print("AVERAGE PERFORMANCE")
print("="*70)
print(avg_comparison.to_string(index=False))

ml_score = comparison["ML Ranker"].mean()
best_baseline = comparison[["Cost Only", "Seasonality Only"]].max(axis=1).mean()
improvement = ((ml_score - best_baseline) / best_baseline) * 100
print(f"\n✓ ML Ranker improves over best baseline by: {improvement:.2f}%")

# ----------------------------
# 17) Statistical significance
# ----------------------------


# ----------------------------
# 18) Feature importance
# ----------------------------
importances = ranker.feature_importances_
feature_importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": importances
}).sort_values("importance", ascending=False)

print("\n=== Feature Importance ===")
print(feature_importance_df)

# ----------------------------
# 19) Visualizations
# ----------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(comparison["Month"], comparison["ML Ranker"], marker='o', linewidth=2, label="ML Ranker")
ax1.plot(comparison["Month"], comparison["Cost Only"], marker='s', linewidth=2, label="Cost Only")
ax1.plot(comparison["Month"], comparison["Seasonality Only"], marker='^', linewidth=2, label="Seasonality Only")

ax1.set_xlabel("Month", fontsize=12)
ax1.set_ylabel("NDCG@10", fontsize=12)
ax1.set_title("Ranking Performance Across Test Months", fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

methods = ["ML Ranker", "Cost Only", "Seasonality Only"]
avg_scores = [
    comparison["ML Ranker"].mean(),
    comparison["Cost Only"].mean(),
    comparison["Seasonality Only"].mean()
]

colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12']
bars = ax2.bar(methods, avg_scores, color=colors, alpha=0.7, edgecolor='black')

for bar in bars:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

ax2.set_ylabel("Average NDCG@10", fontsize=12)
ax2.set_title("Average Performance Comparison", fontsize=14, fontweight='bold')
ax2.set_ylim([0, max(avg_scores) * 1.1])
plt.xticks(rotation=15, ha='right')

plt.tight_layout()
plt.savefig('ranking_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ Visualization saved as 'ranking_comparison.png'")
print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)