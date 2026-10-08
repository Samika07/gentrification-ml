import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve, classification_report, confusion_matrix
)

# ============================================================
# PATHS
# ============================================================
PROCESSED_DIR = Path("data/processed")
RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================
def load_data():
    path = PROCESSED_DIR / "gentrification_dataset.csv"
    if not path.exists():
        raise FileNotFoundError(f"Processed dataset not found at {path}")
    
    df = pd.read_csv(path)
    return df

# ============================================================
# TRAIN & EVALUATE MODELS
# ============================================================
def train_and_evaluate():
    print("\n" + "=" * 70)
    print("LOADING DATA AND PREPARING FOR MODELING")
    print("=" * 70)

    df = load_data()
    
    # Define features and target
    # Exclude TRACT_ID and housing_cost_change_pct (which is perfectly correlated with target)
    features = [
        "population_change_pct",
        "median_age_change",
        "income_change_pct",
        "poverty_change",
        "housing_units_change_pct",
        "rent_change_pct",
        "home_value_change_pct",
        "affordability_pressure"
    ]
    
    X = df[features]
    y = df["gentrification"]
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # Scale Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    
    # Initialize Models
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "Support Vector Machine": SVC(probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
    
    print("\n" + "=" * 70)
    print("TRAINING MODELS")
    print("=" * 70)
    
    results = []
    
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        # Train
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        
        # Evaluate
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_prob)
        
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "ROC-AUC": auc
        })
        
        # Save model
        joblib.dump(model, MODELS_DIR / f"{name.replace(' ', '_').lower()}.pkl")
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
        
        print(f"Results for {name}:")
        print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
        
        # Confusion matrix plot
        plt.figure(figsize=(6,5))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"confusion_matrix_{name.replace(' ', '_').lower()}.png")
        plt.close()
    
    # Save ROC Curve
    plt.figure(1) # Go back to ROC plot
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "roc_curves.png")
    plt.close()
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    
    print("\n" + "=" * 70)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 70)
    print(results_df.to_string(index=False))
    
    # Feature Importance for Random Forest
    rf_model = models["Random Forest"]
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances (Random Forest)")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), np.array(features)[indices], rotation=45, ha="right")
    plt.xlim([-1, X.shape[1]])
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "feature_importances.png")
    plt.close()
    
    print("\nTraining complete! Results and models saved to 'results/' and 'models/' directories.")

if __name__ == "__main__":
    train_and_evaluate()
