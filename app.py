import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Setup Path
MODELS_DIR = Path("models")
model_path = MODELS_DIR / "random_forest.pkl"
scaler_path = MODELS_DIR / "scaler.pkl"

st.set_page_config(page_title="Gentrification Predictor", layout="centered")

st.title("🏙️ Gentrification Prediction App")
st.write("This tool predicts the likelihood of gentrification for a U.S. Census Tract based on its economic and demographic shifts. A tract is considered to be gentrifying if it experiences a **$\ge 10\%$ increase in median gross rent** over a 4-year period.")

st.markdown("---")

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except Exception as e:
    st.error("Model files not found! Please ensure you have run `python src/train_models.py` to generate the models.")
    st.stop()

st.subheader("Tract Demographics & Changes (e.g., 2010 to 2011)")

col1, col2 = st.columns(2)

with col1:
    rent_change_pct = st.number_input("Rent Change (%)", min_value=-100.0, max_value=1000.0, value=2.0, help="Percentage change in median gross rent.")
    income_change_pct = st.number_input("Income Change (%)", min_value=-100.0, max_value=1000.0, value=1.5, help="Percentage change in median household income.")
    affordability_pressure = st.number_input("Affordability Pressure", min_value=0.0, max_value=100.0, value=0.35, help="Median gross rent divided by monthly median income.")
    population_change_pct = st.number_input("Population Change (%)", min_value=-100.0, max_value=1000.0, value=1.0)

with col2:
    home_value_change_pct = st.number_input("Home Value Change (%)", min_value=-100.0, max_value=1000.0, value=3.0)
    median_age_change = st.number_input("Median Age Change (Years)", min_value=-50.0, max_value=50.0, value=0.5)
    poverty_change = st.number_input("Poverty Change (%)", min_value=-100.0, max_value=100.0, value=-0.2, help="Absolute change in poverty percentage.")
    housing_units_change_pct = st.number_input("Housing Units Change (%)", min_value=-100.0, max_value=1000.0, value=0.5)

st.markdown("---")

if st.button("Predict Gentrification Risk", type="primary"):
    # Must match the exact order of features used in training
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
    
    # Scale features
    features_scaled = scaler.transform([features])
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]
    
    if prediction == 1:
        st.error(f"🚨 **High Risk of Gentrification**")
        st.write(f"The model predicts a **{probability:.1%}** probability that rent will increase by $\ge 10\%$ over the subsequent 4 years.")
    else:
        st.success(f"✅ **Low Risk of Gentrification**")
        st.write(f"The model predicts a **{(1 - probability):.1%}** probability that rent will stay relatively stable (increase of $< 10\%$).")
