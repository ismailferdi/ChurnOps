import numpy as np
import pandas as pd
import json
from pathlib import Path
from core.features import NUMERIC_FEATURES

# Note:
# This implementation computes PSI only for numeric features.
# Drift detection for ctegorical features (e.g., using chi-squared or Jensen-Shannon divergence)
# is not implemented nd can can added as fufture extension.


def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:

    _, bin_edges = np.histogram(expected, bins=bins)

    expected_counts, _ = np.histogram(expected, bins=bin_edges)
    actual_counts, _ = np.histogram(actual, bins=bin_edges)

    epsilon = 1e-4

    expected_pct = (expected_counts / expected_counts.sum()) + epsilon
    actual_pct = (actual_counts / actual_counts.sum()) + epsilon

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))

    return float(psi)


def get_psi_status(psi: float) -> str:

    if psi < 0.1:
        return "OK"
    elif 0.1 <= psi < 0.2:
        return "WARNING"
    else:
        return "ALERT"


def load_reference_distributions(path: str) -> dict[str, dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_drift_report(
    df: pd.DataFrame,
    reference_distributions: dict,
) -> list[dict]:
    drift_report = []

    for feature in NUMERIC_FEATURES:
        actual = df[feature].to_numpy()

        reference = np.asarray(reference_distributions[feature]["densities"])

        psi = compute_psi(reference, actual)

        drift_report.append(
            {
                "feature": feature,
                "psi": psi,
                "status": get_psi_status(psi),
            }
        )

    return drift_report
