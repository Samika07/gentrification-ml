import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score

PROCESSED_DIR = Path("data/processed")

def load_data():
    return pd.read_csv(PROCESSED_DIR / "gentrification_dataset.csv")

def extract():
    df = load_data()
    features = [
        "population_change_pct", "median_age_change", "income_change_pct", 
        "poverty_change", "housing_units_change_pct", "rent_change_pct", 
        "home_value_change_pct", "affordability_pressure"
    ]
    X = df[features]
    y = df["gentrification"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    results = []
    
    # Test combinations for Random Forest
    rf_params = [
        {"max_depth": 5, "n_estimators": 50},
        {"max_depth": 10, "n_estimators": 50},
        {"max_depth": 20, "n_estimators": 100},
        {"max_depth": None, "n_estimators": 200},
    ]
    for p in rf_params:
        model = RandomForestClassifier(random_state=42, **p)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        results.append({
            "Model": "Random Forest",
            "Parameters": f"Depth: {p['max_depth']}, Trees: {p['n_estimators']}",
            "Accuracy": accuracy_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred, zero_division=0)
        })

    # Test combinations for XGBoost
    xgb_params = [
        {"learning_rate": 0.01, "max_depth": 3, "n_estimators": 50},
        {"learning_rate": 0.01, "max_depth": 3, "n_estimators": 200},
        {"learning_rate": 0.1, "max_depth": 5, "n_estimators": 100},
        {"learning_rate": 0.2, "max_depth": 7, "n_estimators": 200},
    ]
    for p in xgb_params:
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42, **p)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        results.append({
            "Model": "XGBoost",
            "Parameters": f"LR: {p['learning_rate']}, Depth: {p['max_depth']}, Trees: {p['n_estimators']}",
            "Accuracy": accuracy_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred, zero_division=0)
        })
        
    df_res = pd.DataFrame(results)
    Path("results/tuned").mkdir(parents=True, exist_ok=True)
    df_res.to_csv("results/tuned/hyperparam_details.csv", index=False)
    print(df_res.to_csv(index=False))

if __name__ == "__main__":
    extract()
