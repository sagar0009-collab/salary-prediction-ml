import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Salary_Data.csv"
MODEL_PATH = BASE_DIR / "salary_model.pkl"
METRICS_PATH = BASE_DIR / "metrics.json"
FEATURES_PATH = BASE_DIR / "feature_options.json"

TARGET = "Salary"
# Gender is intentionally excluded from prediction to avoid using a sensitive
# personal attribute as a salary-setting signal.
NUMERIC_FEATURES = ["Age", "Years of Experience"]
CATEGORICAL_FEATURES = ["Education Level", "Job Title"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Normalize common education-level spelling variations.
    if "Education Level" in df.columns:
        education_map = {
            "Bachelor's": "Bachelor's Degree",
            "Master's": "Master's Degree",
            "phD": "PhD",
            "Phd": "PhD",
            "PHD": "PhD",
        }
        df["Education Level"] = df["Education Level"].replace(education_map)

    # Remove exact duplicate records, then rows without a known target.
    df = df.drop_duplicates()
    df = df.dropna(subset=[TARGET])

    # Basic validity checks for numeric fields. Missing feature values are
    # imputed by the sklearn pipeline; impossible negative values are treated
    # as missing rather than inventing replacement values.
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.loc[df["Age"] <= 0, "Age"] = float("nan")
    df.loc[df["Years of Experience"] < 0, "Years of Experience"] = float("nan")
    df = df[df[TARGET] >= 0]
    return df


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Put Salary_Data.csv beside train_model.py."
        )

    raw = pd.read_csv(DATA_PATH)
    cleaned = clean_data(raw)
    X = cleaned[FEATURES]
    y = cleaned[TARGET]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    model = Pipeline([
        ("preprocess", preprocess),
        ("regressor", LinearRegression()),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "rows_in_original_dataset": int(len(raw)),
        "rows_after_cleaning": int(len(cleaned)),
        "exact_duplicates_removed": int(len(raw) - len(raw.drop_duplicates())),
        "features_used": FEATURES,
        "target": TARGET,
        "test_size": 0.20,
        "random_state": 42,
        "mae": round(float(mean_absolute_error(y_test, predictions)), 2),
        "rmse": round(float(mean_squared_error(y_test, predictions) ** 0.5), 2),
        "r2": round(float(r2_score(y_test, predictions)), 4),
        "note": "Metrics are a baseline; repeated records and unusual salary values in the source data may affect results."
    }

    # Retrain final pipeline on all cleaned rows for use in the app.
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)

    options = {
        "education_levels": sorted(cleaned["Education Level"].dropna().astype(str).unique().tolist()),
        "job_titles": sorted(cleaned["Job Title"].dropna().astype(str).unique().tolist()),
        "age_min": int(max(18, cleaned["Age"].min())),
        "age_max": int(cleaned["Age"].max()),
        "experience_min": float(max(0, cleaned["Years of Experience"].min())),
        "experience_max": float(cleaned["Years of Experience"].max()),
    }
    FEATURES_PATH.write_text(json.dumps(options, indent=2), encoding="utf-8")
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Training complete.")
    print(json.dumps(metrics, indent=2))
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved feature options: {FEATURES_PATH}")


if __name__ == "__main__":
    main()
