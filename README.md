# Gentrification Prediction using Machine Learning

## Project Overview

This project aims to predict gentrification at the U.S. Census tract level using demographic, economic, and housing-related data from the U.S. Census American Community Survey (ACS).

The machine learning pipeline currently covers:

1. Census data collection
2. Data inspection
3. Data cleaning
4. Feature engineering
5. Creation of the gentrification target
6. Generation of an ML-ready dataset
7. Model training and evaluation (Logistic Regression, SVM, Random Forest, XGBoost)

---

## Dataset

The project uses California Census tract-level ACS 5-Year data for:
- 2010
- 2011
- 2012
- 2016

Raw datasets are stored in `data/raw/` and processed feature datasets in `data/processed/`.

---

## Setup Instructions

### Prerequisites
- Python 3.8+
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Samika07/gentrification-ml
   cd gentrification-ml
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   # On macOS / Linux
   python -m venv venv
   source venv/bin/activate
   
   # On Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Project

### 1. Data Preprocessing
Generate the final ML-ready dataset (`gentrification_dataset.csv`) by running the preprocessing script:
```bash
python src/preprocessing.py
```

### 2. Model Training and Evaluation
Train the classification models and generate evaluation metrics, plots, and saved model files:
```bash
python src/train_models.py
```
This will create a `results/` folder with ROC curves, feature importances, confusion matrices, and the test set performance summary, as well as a `models/` folder containing the saved `.pkl` scaler and model weights.
