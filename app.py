import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "salary_model.pkl"
OPTIONS_PATH = BASE_DIR / "feature_options.json"
METRICS_PATH = BASE_DIR / "metrics.json"

st.set_page_config(page_title="Salary Prediction", page_icon="💼", layout="centered")

st.title("💼 Salary Prediction")
st.write(
    "Estimate a salary using a Multiple Linear Regression model trained on "
    "age, education, job title, and years of experience."
)
st.caption(
    "This is an educational estimate, not a guarantee of actual pay. "
    "Predictions may be inaccurate for unusual roles or salary ranges."
)

if not MODEL_PATH.exists() or not OPTIONS_PATH.exists():
    st.warning("Model files are missing. Run `python train_model.py` first.")
    st.stop()

model = joblib.load(MODEL_PATH)
options = json.loads(OPTIONS_PATH.read_text(encoding="utf-8"))

with st.form("salary_form"):
    st.subheader("Enter candidate details")
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=max(80, int(options["age_max"])),
        value=30,
        step=1,
    )
    experience_max = max(40.0, float(options["experience_max"]))
    experience = st.number_input(
        "Years of Experience",
        min_value=0.0,
        max_value=experience_max,
        value=3.0,
        step=0.5,
    )
    education = st.selectbox("Education Level", options["education_levels"])
    job_title = st.selectbox("Job Title", options["job_titles"])
    submitted = st.form_submit_button("Predict Salary", type="primary", use_container_width=True)

if submitted:
    input_df = pd.DataFrame([{
        "Age": age,
        "Years of Experience": experience,
        "Education Level": education,
        "Job Title": job_title,
    }])
    prediction = float(model.predict(input_df)[0])
    st.success(f"Estimated salary: {prediction:,.0f} (dataset currency/units)")
    st.caption(
        "The dataset does not specify a currency, so confirm the source before "
        "interpreting this number as INR, USD, or another currency."
    )

if METRICS_PATH.exists():
    with st.expander("About model evaluation"):
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f'{metrics["mae"]:,.0f}')
        c2.metric("RMSE", f'{metrics["rmse"]:,.0f}')
        c3.metric("R²", f'{metrics["r2"]:.3f}')
        st.write(
            f'Cleaned rows: {metrics["rows_after_cleaning"]:,} '
            f'of {metrics["rows_in_original_dataset"]:,}.'
        )
        st.caption(
            "MAE and RMSE are in the dataset's salary units. R² is not a percentage "
            "and can be negative on a poor model. Metrics are a baseline, not a promise."
        )
