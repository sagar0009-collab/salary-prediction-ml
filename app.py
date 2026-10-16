
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="Salary Prediction ML",
    page_icon="💼",
    layout="wide"
)

st.title("💼 Salary Prediction System")
st.write("Predict estimated salary using Multiple Linear Regression.")

@st.cache_data
def load_data():
    df = pd.read_csv("Salary_Data.csv")

    required = [
        "Age",
        "Education Level",
        "Job Title",
        "Years of Experience",
        "Salary"
    ]

    df = df[required].copy()

    for col in ["Age", "Years of Experience", "Salary"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.drop_duplicates()
    df = df.dropna(subset=["Salary"])
    df = df[df["Salary"] >= 0]

    return df


@st.cache_resource
def train_model():
    df = load_data()

    features = [
        "Age",
        "Education Level",
        "Job Title",
        "Years of Experience"
    ]

    X = df[features]
    y = df["Salary"]

    numeric_features = ["Age", "Years of Experience"]
    categorical_features = ["Education Level", "Job Title"]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
        "R2": r2_score(y_test, predictions)
    }

    return model, metrics, df


try:
    model, metrics, df = train_model()
except Exception as e:
    st.error(f"Could not load or train the model: {e}")
    st.info("Ensure Salary_Data.csv is in the same GitHub folder as app.py.")
    st.stop()


st.subheader("📊 Model Performance")

c1, c2, c3 = st.columns(3)

c1.metric("MAE", f"{metrics['MAE']:,.0f}")
c2.metric("RMSE", f"{metrics['RMSE']:,.0f}")
c3.metric("R² Score", f"{metrics['R2']:.3f}")

st.caption(
    "Metrics are calculated on a held-out test set. "
    "Salary units follow the original dataset."
)

st.divider()
st.subheader("Enter Candidate Details")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=25
    )

    education_options = sorted(
        df["Education Level"].dropna().astype(str).unique()
    )

    education = st.selectbox(
        "Education Level",
        education_options
    )

with col2:
    experience_max = max(
        1.0,
        float(df["Years of Experience"].max())
    )

    experience = st.number_input(
        "Years of Experience",
        min_value=0.0,
        max_value=experience_max,
        value=2.0,
        step=0.5
    )

    job_options = sorted(
        df["Job Title"].dropna().astype(str).unique()
    )

    job = st.selectbox("Job Title", job_options)


if st.button("Predict Salary", type="primary"):
    input_data = pd.DataFrame([{
        "Age": age,
        "Education Level": education,
        "Job Title": job,
        "Years of Experience": experience
    }])

    prediction = model.predict(input_data)[0]

    st.success(f"Estimated Salary: {prediction:,.2f}")

    st.caption(
        "This is a statistical estimate, not a guaranteed salary offer. "
        "Results depend on the quality and coverage of the training data."
    )

with st.expander("Preview Dataset"):
    st.dataframe(df.head(20), use_container_width=True)
    st.write(f"Records used: {len(df):,}")
