from api.core.errors import DriftReportError
import pandas as pd
import sqlite3
from monitoring.drift_detector import compute_drift_report, load_reference_distributions
from api.core.config import settings


def load_prediction_log(db_path: str, window: int) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:

        count = conn.execute("SELECT COUNT(*) FROM predicions;").fetchone()[0]

        if count == 0:
            return pd.DataFrame()

        df = pd.read_sql_query(
            """
            SELECT *
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            conn,
            params=(window,),
        )

    return df.iloc[::-1].reset_index(drop=True)


def get_drift_report(db_path: str, window: int) -> dict:
    try:
        df = load_prediction_log(db_path=db_path, window=window)

        if len(df) == 0:
            return {
                "total_predictions": 0,
                "window_size": window,
                "drift_metrics": [],
            }

        reference_distributions = load_reference_distributions(
            settings.reference_distributions_path
        )

        drift_metrics = compute_drift_report(
            df,
            reference_distributions,
        )

        return {
            "total_predictions": len(df),
            "window_size": window,
            "drift_metrics": drift_metrics,
        }

    except sqlite3.DatabaseError as exc:
        raise DriftReportError(
            client_message="The SQLite database is corrupt or unreadable."
        ) from exc
