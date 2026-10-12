import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Setup Paths
MODELS_DIR = Path("models")
TUNED_MODELS_DIR = Path("models/tuned")

st.set_page_config(page_title="Gentrification Predictor", page_icon="🏙️", layout="wide")

st.title("🏙️ Gentrification Prediction & Risk Assessment")
st.write("This application predicts whether a U.S. Census tract in California is likely to gentrify over a 4-year horizon (defined as a **$\ge 10\%$ increase in median gross rent**).")

# =========================================================
# SIDEBAR: MODEL & TUNING SELECTION + PRESETS
# =========================================================
st.sidebar.header("⚙️ Model & Hyperparameter Tuning")

model_choice = st.sidebar.selectbox(
    "Select Model & Tuning Configuration:",
    [
        "Tuned XGBoost (Best Overall - AUC 0.683)",
        "Tuned Random Forest (Balanced - Depth 10)",
        "Tuned SVM (High Recall - Balanced Weights)",
        "Baseline Random Forest (Default)",
        "Baseline Logistic Regression"
    ]
)

# Map choice to file paths
model_map = {
    "Tuned XGBoost (Best Overall - AUC 0.683)": (TUNED_MODELS_DIR / "xgboost_tuned.pkl", TUNED_MODELS_DIR / "scaler.pkl"),
    "Tuned Random Forest (Balanced - Depth 10)": (TUNED_MODELS_DIR / "random_forest_tuned.pkl", TUNED_MODELS_DIR / "scaler.pkl"),
    "Tuned SVM (High Recall - Balanced Weights)": (TUNED_MODELS_DIR / "support_vector_machine_tuned.pkl", TUNED_MODELS_DIR / "scaler.pkl"),
    "Baseline Random Forest (Default)": (MODELS_DIR / "random_forest.pkl", MODELS_DIR / "scaler.pkl"),
    "Baseline Logistic Regression": (MODELS_DIR / "logistic_regression.pkl", MODELS_DIR / "scaler.pkl")
}

model_path, scaler_path = model_map[model_choice]

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except Exception as e:
    st.error(f"Error loading model from {model_path}: {e}")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.header("🎯 Demo Quick Scenarios")
st.sidebar.write("Click a preset to quickly populate feature values for demoing:")

# Initialize session state for inputs if not present
defaults = {
    "rent_change_pct": 2.0,
    "income_change_pct": 1.5,
    "affordability_pressure": 0.35,
    "population_change_pct": 1.0,
    "home_value_change_pct": 3.0,
    "median_age_change": 0.5,
    "poverty_change": -0.2,
    "housing_units_change_pct": 0.5
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if st.sidebar.button("🚨 Scenario 1: Rapid Gentrification (High Risk)"):
    st.session_state["rent_change_pct"] = 14.5
    st.session_state["income_change_pct"] = 1.0
    st.session_state["affordability_pressure"] = 0.52
    st.session_state["population_change_pct"] = 5.2
    st.session_state["home_value_change_pct"] = 22.0
    st.session_state["median_age_change"] = -1.8
    st.session_state["poverty_change"] = -3.5
    st.session_state["housing_units_change_pct"] = 3.0

if st.sidebar.button("✅ Scenario 2: Stable Community (Low Risk)"):
    st.session_state["rent_change_pct"] = 0.8
    st.session_state["income_change_pct"] = 3.5
    st.session_state["affordability_pressure"] = 0.22
    st.session_state["population_change_pct"] = 0.4
    st.session_state["home_value_change_pct"] = 1.5
    st.session_state["median_age_change"] = 0.2
    st.session_state["poverty_change"] = -0.5
    st.session_state["housing_units_change_pct"] = 0.8

if st.sidebar.button("⚠️ Scenario 3: Borderline / Emerging Pressure"):
    st.session_state["rent_change_pct"] = 6.5
    st.session_state["income_change_pct"] = 2.0
    st.session_state["affordability_pressure"] = 0.38
    st.session_state["population_change_pct"] = 2.0
    st.session_state["home_value_change_pct"] = 9.0
    st.session_state["median_age_change"] = -0.4
    st.session_state["poverty_change"] = -1.0
    st.session_state["housing_units_change_pct"] = 1.2

# =========================================================
# MAIN PANEL: INPUT FIELDS
# =========================================================
st.subheader("Census Tract Momentum & Economic Indicators (1-Year Change)")

col1, col2 = st.columns(2)

with col1:
    rent_change_pct = st.number_input(
        "Rent Change (%)", 
        min_value=-100.0, max_value=1000.0, 
        value=float(st.session_state["rent_change_pct"]), 
        help="1-year percentage change in median gross rent."
    )
    income_change_pct = st.number_input(
        "Household Income Change (%)", 
        min_value=-100.0, max_value=1000.0, 
        value=float(st.session_state["income_change_pct"]), 
        help="1-year percentage change in median household income."
    )
    affordability_pressure = st.number_input(
        "Affordability Pressure (Rent / Monthly Income)", 
        min_value=0.0, max_value=10.0, 
        value=float(st.session_state["affordability_pressure"]), 
        help="Monthly gross rent divided by monthly median income. Values > 0.30 indicate rent-burdened communities."
    )
    population_change_pct = st.number_input(
        "Population Change (%)", 
        min_value=-100.0, max_value=1000.0, 
        value=float(st.session_state["population_change_pct"])
    )

with col2:
    home_value_change_pct = st.number_input(
        "Home Value Change (%)", 
        min_value=-100.0, max_value=1000.0, 
        value=float(st.session_state["home_value_change_pct"]),
        help="Capital appreciation indicator: 1-year change in median home property values."
    )
    median_age_change = st.number_input(
        "Median Age Change (Years)", 
        min_value=-50.0, max_value=50.0, 
        value=float(st.session_state["median_age_change"]),
        help="Shift in median resident age (negative values indicate influx of younger demographics)."
    )
    poverty_change = st.number_input(
        "Poverty Rate Shift (Percentage Points)", 
        min_value=-100.0, max_value=100.0, 
        value=float(st.session_state["poverty_change"]), 
        help="Absolute percentage point change in poverty rate."
    )
    housing_units_change_pct = st.number_input(
        "Housing Units Change (%)", 
        min_value=-100.0, max_value=1000.0, 
        value=float(st.session_state["housing_units_change_pct"]),
        help="Supply growth rate of total residential units."
    )

st.markdown("---")

# =========================================================
# PREDICTION & OUTPUT GAUGES
# =========================================================
if st.button("🚀 Predict Gentrification Risk", type="primary", use_container_width=True):
    # Order must strictly match training feature array
    features = [
        population_change_pct,
        median_age_change,
        income_change_pct,
        poverty_change,
        housing_units_change_pct,
        rent_change_pct,
        home_value_change_pct,
        affordability_pressure
    ]
    
    features_scaled = scaler.transform([features])
    
    prediction = int(model.predict(features_scaled)[0])
    probabilities = model.predict_proba(features_scaled)[0]
    prob_gentrify = probabilities[1]
    prob_stable = probabilities[0]
    
    st.subheader("📊 Model Assessment & Risk Gauge")
    
    gauge_col1, gauge_col2, gauge_col3 = st.columns([1, 2, 1])
    
    with gauge_col2:
        if prediction == 1 or prob_gentrify >= 0.50:
            st.error(f"🚨 **PREDICTION: HIGH GENTRIFICATION RISK**")
            st.metric(label="Gentrification Probability (Rent increase ≥ 10%)", value=f"{prob_gentrify * 100:.1f}%")
            st.progress(float(prob_gentrify))
            st.markdown(
                f"""
                **Analysis Summary:**
                * **Active Model:** `{model_choice}`
                * The model calculates an estimated **{prob_gentrify * 100:.1f}%** chance that median gross rent will increase by **$\ge 10\%$** over the subsequent 4 years.
                * **Primary Risk Drivers:** Rapid property appreciation (`home_value_change_pct`), high rent momentum (`rent_change_pct`), and acute affordability strain (`affordability_pressure` = {affordability_pressure:.2f}).
                """
            )
        else:
            st.success(f"✅ **PREDICTION: LOW GENTRIFICATION RISK**")
            st.metric(label="Stability Probability (Rent increase < 10%)", value=f"{prob_stable * 100:.1f}%")
            st.progress(float(prob_gentrify))
            st.markdown(
                f"""
                **Analysis Summary:**
                * **Active Model:** `{model_choice}`
                * The model estimates an **{prob_stable * 100:.1f}%** chance that rent will remain stable or increase moderately ($< 10\%$).
                * **Stabilizing Factors:** Household income growth keeps pace with housing expenses, and affordability pressure remains within sustainable ranges.
                """
            )
