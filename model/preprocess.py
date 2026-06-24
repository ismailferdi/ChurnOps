import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from core.features import (
    TARGET,
    NUMERIC_FEATURES,
    BINARY_CATEGORICAL_FEATURES,
    MULTI_CATEGORICAL_FEATURES,
    PASSTHROUGH_FEATURES,
)
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

DEFAULT_PIPELINE_PATH = Path(__file__).parent / "pipeline.pkl"

data_transformer = ColumnTransformer(
    transformers=[
        ("num_features", StandardScaler(), NUMERIC_FEATURES),
        (
            "bin_cat_features",
            OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
            BINARY_CATEGORICAL_FEATURES,
        ),
        (
            "mul_cat_features",
            OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
            MULTI_CATEGORICAL_FEATURES,
        ),
        ("keep", "passthrough", PASSTHROUGH_FEATURES),
    ]
)

data_pipeline = Pipeline([("preprocessor", data_transformer)])


def load_data(path: str) -> pd.DataFrame:

    if not path or not Path(path).exists():
        return pd.DataFrame()

    df = pd.read_csv(path)

    if "customerID" in df.columns:
        df = df.drop("customerID", axis=1)

    df["TotalCharges"] = df["TotalCharges"].replace(" ", "NaN").astype(float)
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    df[TARGET] = (df[TARGET] == "Yes").astype(int)

    return df


def split_data(df: pd.DataFrame) -> tuple:

    X = df.drop(TARGET, axis=1)
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test


def fit_and_save_preprocessor(X_train: pd.DataFrame, path: str) -> Pipeline:

    fited_pipeline = data_pipeline.fit(X_train)
    if path:
        joblib.dump(fited_pipeline, path)
    else:
        joblib.dump(fited_pipeline, DEFAULT_PIPELINE_PATH)

    return fited_pipeline


def load_preprocessor(path: str) -> Pipeline:

    try:
        pipeline = joblib.load(path)
        return pipeline
    except FileNotFoundError:
        raise FileNotFoundError("This pipline doesn't exist.")


def transform(preprocessor: Pipeline, X: pd.DataFrame) -> np.ndarray:

    return preprocessor.transform(X)
