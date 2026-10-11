
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    classification_report
)

# =========================================================
# 1. LOAD DATA
# =========================================================

DATA_PATH = "data/processed/gentrification_dataset.csv"
OUTPUT_DIR = "results/shap"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

FEATURES = [
    "population_change_pct",
    "median_age_change",
    "income_change_pct",
    "poverty_change",
    "housing_units_change_pct",
    "rent_change_pct",
    "home_value_change_pct",
    "affordability_pressure"
]

TARGET = "gentrification"

# Do not use housing_cost_change_pct as an input:
# it was used to construct the target label.
df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna(subset=FEATURES + [TARGET])

X = df[FEATURES]
y = df[TARGET].astype(int)

print("Dataset shape:", df.shape)
print("Target distribution:")
print(y.value_counts())

# =========================================================
# 2. TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# =========================================================
# 3. TRAIN XGBOOST CLASSIFIER
# =========================================================

model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

print("\nTraining XGBoost...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("\nMODEL PERFORMANCE")
print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
print("F1 score:", round(f1_score(y_test, predictions), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, probabilities), 4))
print(classification_report(y_test, predictions))

# =========================================================
# 4. CALCULATE SHAP VALUES
# =========================================================

print("\nCalculating SHAP values...")

# Explain a sample of test records to keep plots readable
X_explain = X_test.sample(
    n=min(500, len(X_test)),
    random_state=42
)

explainer = shap.TreeExplainer(model)
shap_values = explainer(X_explain)

# =========================================================
# 5. GLOBAL FEATURE IMPORTANCE: BAR PLOT
# =========================================================

plt.figure()
shap.plots.bar(shap_values, max_display=8, show=False)
plt.tight_layout()
plt.savefig(
    f"{OUTPUT_DIR}/shap_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# =========================================================
# 6. SHAP SUMMARY / BEESWARM PLOT
# =========================================================

plt.figure()
shap.plots.beeswarm(shap_values, max_display=8, show=False)
plt.tight_layout()
plt.savefig(
    f"{OUTPUT_DIR}/shap_summary.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# =========================================================
# 7. EXPLAIN ONE INDIVIDUAL PREDICTION
# =========================================================

# Explain the first sampled test record
one_explanation = shap_values[0]

plt.figure()
shap.plots.waterfall(one_explanation, max_display=8, show=False)
plt.tight_layout()
plt.savefig(
    f"{OUTPUT_DIR}/shap_individual_prediction.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()

# =========================================================
# 8. EXPORT GLOBAL IMPORTANCE NUMBERS
# =========================================================

importance = pd.DataFrame({
    "Feature": FEATURES,
    "Mean_Absolute_SHAP": np.abs(
        shap_values.values
    ).mean(axis=0)
})

importance = importance.sort_values(
    "Mean_Absolute_SHAP",
    ascending=False
)

importance.to_csv(
    f"{OUTPUT_DIR}/shap_feature_importance.csv",
    index=False
)

print("\nGLOBAL SHAP IMPORTANCE")
print(importance.to_string(index=False))

print("\nSaved SHAP outputs to:", OUTPUT_DIR)
print(" - shap_feature_importance.png")
print(" - shap_summary.png")
print(" - shap_individual_prediction.png")
print(" - shap_feature_importance.csv")
