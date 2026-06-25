from core.features import FEATURE_ORDER, NUMERIC_FEATURES
from api.models.requests import CustomerFeatures
import logging
import sqlite3
import datetime


def build_predictions_table_sql() -> str:
    columns = [
        "id INTEGER PRIMARY KEY AUTOINCREMENT",
        "timestamp TEXT",
        "churn_prediction INTEGER",
        "churn_probability REAL",
    ]

    for feature in FEATURE_ORDER:
        column_type = "REAL" if feature in NUMERIC_FEATURES else "TEXT"
        columns.append(f'"{feature}" {column_type}')

    return f"CREATE TABLE IF NOT EXISTS predictions " f"({', '.join(columns)});"


def timestamp() -> str:
    return datetime.datetime.now().isoformat()


def inseart_prediction_raw_sql(customer: CustomerFeatures, result: dict) -> str:
    features = customer.to_dataframe()
    first_line = "INSERT INTO predictions (timestamp,churn_prediction,churn_probability"
    second_line = (
        f"VALUES ('{timestamp()}',"
        f"{result['churn_prediction']},"
        f"{result['churn_probability']}"
    )
    for feature in FEATURE_ORDER:
        first_line += f",{feature}"
        value = features[feature].iloc[0]
        if feature in NUMERIC_FEATURES:
            second_line += f",{value}"
        else:
            second_line += f",'{value}'"
    first_line += ")"
    second_line += ");"
    return first_line + "\n" + second_line


def initialize_log_db(db_path: str):
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute(build_predictions_table_sql())
    except Exception as e:
        logging.warning(f"Failed to create the table.\nCause: {e}", exc_info=True)


def log_prediction(db_path: str, customer: CustomerFeatures, result: dict):
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute(inseart_prediction_raw_sql(customer, result))
    except Exception as e:
        logging.warning(f"Failed to log prediction.\nCause: {e}", exc_info=True)
