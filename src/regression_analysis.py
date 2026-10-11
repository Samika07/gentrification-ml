import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# 1. LOAD DATA
# ============================================================

DATA_PATH = "data/processed/gentrification_dataset.csv"

print("=" * 70)
print("LOADING PROCESSED DATA")
print("=" * 70)

df = pd.read_csv(DATA_PATH)

print(f"Original dataset shape: {df.shape}")


# ============================================================
# 2. DEFINE FEATURES AND TARGET
# ============================================================

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

TARGET = "housing_cost_change_pct"


# ============================================================
# 3. CLEAN TARGET FOR REGRESSION ONLY
# ============================================================

print("\n" + "=" * 70)
print("CLEANING REGRESSION TARGET")
print("=" * 70)

print("\nOriginal target statistics:")
print(df[TARGET].describe())

# The Census data contains suppressed/invalid values which can
# create astronomically large percentage changes.
#
# For this regression experiment we restrict the target to a
# reasonable range of housing-cost percentage changes.

LOWER_LIMIT = -100
UPPER_LIMIT = 500

before = len(df)

df_reg = df[
    (df[TARGET] >= LOWER_LIMIT) &
    (df[TARGET] <= UPPER_LIMIT)
].copy()

after = len(df_reg)

removed = before - after

print(f"\nRows before cleaning: {before}")
print(f"Rows removed:        {removed}")
print(f"Rows remaining:      {after}")

print("\nCleaned target statistics:")
print(df_reg[TARGET].describe())


# ============================================================
# 4. REMOVE MISSING / INFINITE VALUES
# ============================================================

df_reg = df_reg.replace(
    [np.inf, -np.inf],
    np.nan
)

df_reg = df_reg.dropna(
    subset=FEATURES + [TARGET]
)

print(f"\nDataset after final cleaning: {df_reg.shape}")


# ============================================================
# 5. DEFINE X AND Y
# ============================================================

X = df_reg[FEATURES]
y = df_reg[TARGET]

print("\nFeatures:")
for feature in FEATURES:
    print(f"  - {feature}")

print("\nRegression target:")
print(f"  - {TARGET}")


# ============================================================
# 6. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")


# ============================================================
# 7. LINEAR REGRESSION
# ============================================================

print("\n" + "=" * 70)
print("LINEAR REGRESSION")
print("=" * 70)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

linear_model = LinearRegression()

linear_model.fit(
    X_train_scaled,
    y_train
)

linear_predictions = linear_model.predict(
    X_test_scaled
)


# ============================================================
# 8. LINEAR REGRESSION METRICS
# ============================================================

linear_mae = mean_absolute_error(
    y_test,
    linear_predictions
)

linear_mse = mean_squared_error(
    y_test,
    linear_predictions
)

linear_rmse = np.sqrt(linear_mse)

linear_r2 = r2_score(
    y_test,
    linear_predictions
)

print(f"MAE:  {linear_mae:.4f}")
print(f"RMSE: {linear_rmse:.4f}")
print(f"R²:   {linear_r2:.4f}")


# ============================================================
# 9. LINEAR REGRESSION COEFFICIENTS
# ============================================================

print("\n" + "=" * 70)
print("LINEAR REGRESSION COEFFICIENTS")
print("=" * 70)

coefficients = pd.DataFrame({
    "Feature": FEATURES,
    "Coefficient": linear_model.coef_
})

coefficients["Absolute_Coefficient"] = (
    coefficients["Coefficient"].abs()
)

coefficients = coefficients.sort_values(
    "Absolute_Coefficient",
    ascending=False
)

print(coefficients.to_string(index=False))


# ============================================================
# 10. RANDOM FOREST REGRESSOR
# ============================================================

print("\n" + "=" * 70)
print("RANDOM FOREST REGRESSOR")
print("=" * 70)

rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(
    X_train,
    y_train
)

rf_predictions = rf_model.predict(
    X_test
)


# ============================================================
# 11. RANDOM FOREST METRICS
# ============================================================

rf_mae = mean_absolute_error(
    y_test,
    rf_predictions
)

rf_mse = mean_squared_error(
    y_test,
    rf_predictions
)

rf_rmse = np.sqrt(rf_mse)

rf_r2 = r2_score(
    y_test,
    rf_predictions
)

print(f"MAE:  {rf_mae:.4f}")
print(f"RMSE: {rf_rmse:.4f}")
print(f"R²:   {rf_r2:.4f}")


# ============================================================
# 12. MODEL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("REGRESSION MODEL COMPARISON")
print("=" * 70)

comparison = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest Regressor"
    ],
    "MAE": [
        linear_mae,
        rf_mae
    ],
    "RMSE": [
        linear_rmse,
        rf_rmse
    ],
    "R2": [
        linear_r2,
        rf_r2
    ]
})

print(comparison.to_string(index=False))


# ============================================================
# 13. SAVE RESULTS
# ============================================================

os.makedirs(
    "results",
    exist_ok=True
)

comparison.to_csv(
    "results/regression_comparison.csv",
    index=False
)

coefficients.to_csv(
    "results/linear_regression_coefficients.csv",
    index=False
)


# ============================================================
# 14. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("REGRESSION ANALYSIS COMPLETE")
print("=" * 70)

print("\nThis experiment predicts continuous")
print("housing_cost_change_pct instead of binary gentrification.")

print("\nRegression dataset:")
print(f"  Original rows : {before}")
print(f"  Removed       : {removed}")
print(f"  Final rows    : {after}")

print("\nResults saved to:")
print("  results/regression_comparison.csv")
print("  results/linear_regression_coefficients.csv")