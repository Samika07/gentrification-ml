# Gentrification Prediction using Machine Learning

## Project Overview

This project predicts gentrification at the U.S. Census tract level across California using demographic, economic, and housing-related data from the U.S. Census American Community Survey (ACS). 

The full machine learning pipeline covers:

1. **Census Data Collection & Inspection:** ACS 5-Year estimates (2010, 2011, 2012, 2016).
2. **Data Cleaning & Harmonization:** Deduplicating Census tracts and addressing missing/invalid records.
3. **Feature Engineering:** Temporal percentage changes in demographic, economic, and housing indicators, plus custom domain features (e.g., `affordability_pressure`).
4. **Target Formulation:**
   - **Binary Classification:** Tracts with a $\ge 10\%$ increase in median gross rent between 2012 and 2016.
   - **Continuous Regression:** Direct prediction of 4-year rent growth percentage (`housing_cost_change_pct`).
5. **Model Training & Comparison:**
   - Classification: Logistic Regression, SVM, Random Forest, XGBoost.
   - Regression: Linear Regression, Random Forest Regressor.
6. **Hyperparameter Tuning:** Systematic Grid Search with cross-validation across depth, learning rate, tree count, and class weights.
7. **Explainable AI (XAI / SHAP):** Unpacking model decisions using TreeExplainer, beeswarm summary plots, and local prediction waterfalls.
8. **Interactive Web Application:** A Streamlit interface for real-time scenario testing and risk inference.

---

## Dataset

The project uses California tract-level ACS 5-Year data:
- `2010` & `2011`: Baseline predictors and historical change rates
- `2012` & `2016`: Target calculation period (4-year horizon)

Directory structure:
- `data/raw/`: Raw CSV files per census year (`ca_2010.csv`, `ca_2011.csv`, `ca_2012.csv`, `ca_2016.csv`)
- `data/processed/`: Cleaned, feature-engineered dataset (`gentrification_dataset.csv`)

---

## Project Structure

```text
├── data/
│   ├── processed/
│   │   └── gentrification_dataset.csv
│   └── raw/
├── models/
│   ├── scaler.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   └── tuned/
├── results/
│   ├── model_comparison.csv
│   ├── roc_curves.png
│   ├── feature_importances.png
│   ├── linear_regression_coefficients.csv
│   ├── regression_comparison.csv
│   ├── shap/
│   │   ├── shap_feature_importance.csv
│   │   ├── shap_feature_importance.png
│   │   ├── shap_summary.png
│   │   └── shap_individual_prediction.png
│   └── tuned/
│       ├── tuned_model_comparison.csv
│       └── hyperparam_details.csv
├── src/
│   ├── download_census.py
│   ├── inspect_data.py
│   ├── preprocessing.py
│   ├── train_models.py
│   ├── tune_hyperparameters.py
│   ├── extract_grid_details.py
│   ├── regression_analysis.py
│   └── shap_analysis.py
├── app.py
├── requirements.txt
└── README.md
```

---

## Setup Instructions

### Prerequisites
- Python 3.8+
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Samika07/gentrification-ml.git
   cd gentrification-ml
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # On macOS / Linux:
   python -m venv venv
   source venv/bin/activate

   # On Windows:
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Run

### 1. Data Preprocessing
Generate the feature-engineered dataset from raw ACS tables:
```bash
python src/preprocessing.py
```

### 2. Baseline Model Training & Evaluation
Train baseline classifiers (Logistic Regression, SVM, Random Forest, XGBoost), generate confusion matrices, and plot ROC curves:
```bash
python src/train_models.py
```

### 3. Hyperparameter Tuning
Run Grid Search Cross-Validation across parameter spaces to find optimal configurations:
```bash
python src/tune_hyperparameters.py
```
To view specific parameter comparisons and trade-offs:
```bash
python src/extract_grid_details.py
```

### 4. Continuous Regression Analysis
Evaluate continuous rent growth prediction (Linear Regression vs. Random Forest Regressor):
```bash
python src/regression_analysis.py
```

### 5. Model Explainability with SHAP
Compute Shapley values and generate global summary and local waterfall plots:
```bash
python src/shap_analysis.py
```

### 6. Interactive Web Application
Launch the Streamlit dashboard for real-time predictions:
```bash
streamlit run app.py
```

---

## Key Findings & Results Summary

### 1. Classification & Hyperparameter Tuning
Tree-based architectures significantly outperformed linear baselines in handling noisy, non-linear census dynamics:

| Model | Baseline Accuracy | Tuned Accuracy | Tuned ROC-AUC | Optimal Hyperparameters |
| :--- | :---: | :---: | :---: | :--- |
| **XGBoost** | 61.82% | **65.45%** | **0.6834** | `learning_rate=0.01`, `max_depth=3`, `n_estimators=200` |
| **Random Forest** | 62.32% | **64.51%** | **0.6568** | `max_depth=10`, `n_estimators=50`, `min_samples_split=5` |
| **SVM** | 58.37% | 48.59% | 0.5453 | `C=10`, `kernel='rbf'`, `class_weight='balanced'` (Recall: 72.7%) |
| **Logistic Regression** | 58.18% | 58.18% | 0.5485 | `C=10.0`, `solver='liblinear'` |

### 2. Continuous Regression
- **Linear Regression:** $R^2 = -0.0520$ (indicating linear models fail to outperform a baseline mean prediction).
- **Random Forest Regressor:** $R^2 = 0.0780$, $\text{MAE} = 11.33\%$.

### 3. SHAP Explainability Insights
- **Top Predictor:** `affordability_pressure` (Mean |SHAP| = 0.465) is the single most critical driver of impending gentrification.
- **Secondary Drivers:** `rent_change_pct` (0.270) and `home_value_change_pct` (0.200) provide strong momentum signals.
