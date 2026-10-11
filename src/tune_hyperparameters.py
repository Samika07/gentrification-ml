import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Paths
PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results/tuned")
MODELS_DIR = Path("models/tuned")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    path = PROCESSED_DIR / "gentrification_dataset.csv"
    return pd.read_csv(path)

def tune_models():
    print("=" * 70)
    print("HYPERPARAMETER TUNING")
    print("=" * 70)
    
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
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

    # Define hyperparameter grids
    param_grids = {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "params": {
                "C": [0.01, 0.1, 1.0, 10.0],
                "solver": ["lbfgs", "liblinear"],
                "class_weight": [None, "balanced"]
            }
        },
        "Support Vector Machine": {
            "model": SVC(probability=True, random_state=42),
            "params": {
                "C": [0.1, 1, 10],
                "kernel": ["linear", "rbf"],
                "class_weight": [None, "balanced"]
            }
        },
        "Random Forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5],
                "class_weight": [None, "balanced"]
            }
        },
        "XGBoost": {
            "model": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
            "params": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7],
                "scale_pos_weight": [1, 2] # equivalent to balanced weights
            }
        }
    }

    results = []
    
    for name, config in param_grids.items():
        print(f"\nTuning {name}...")
        # Using Grid Search with Cross-Validation
        grid = GridSearchCV(config["model"], config["params"], cv=3, scoring='roc_auc', n_jobs=2)
        grid.fit(X_train_scaled, y_train)
        
        best_model = grid.best_estimator_
        print(f"Best Parameters: {grid.best_params_}")
        
        # Predict on Test Set
        y_pred = best_model.predict(X_test_scaled)
        y_prob = best_model.predict_proba(X_test_scaled)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        
        results.append({
            "Model": name,
            "Best Params": str(grid.best_params_),
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "ROC-AUC": auc
        })
        
        joblib.dump(best_model, MODELS_DIR / f"{name.replace(' ', '_').lower()}_tuned.pkl")

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_DIR / "tuned_model_comparison.csv", index=False)
    
    print("\n" + "=" * 70)
    print("TUNED MODEL COMPARISON")
    print("=" * 70)
    print(results_df[["Model", "Accuracy", "ROC-AUC", "F1 Score"]].to_string(index=False))

if __name__ == "__main__":
    tune_models()
