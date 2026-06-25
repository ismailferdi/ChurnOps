from pydantic import BaseModel, Field, field_validator
from typing import Literal, List


class PredictionResponse(BaseModel):

    churn_prediction: Literal[0, 1]
    churn_probability: float = Field(ge=0.0, le=1.0)
    prediction_label: Literal["Will Not Churn", "Will Churn"]

    @field_validator("churn_probability")
    @classmethod
    def round_probability(cls, v):
        return round(v, 4)


class HelthResponse(BaseModel):

    status: str
    model_loaded: bool
    preprocessor_loaded: bool


class DriftMetric(BaseModel):

    feature: str
    psi: float
    status: Literal["OK", "WARNING", "ALERT"]


class MetricsResponse(BaseModel):

    total_predictions: int
    window_size: int
    drift_metrics: List[DriftMetric]
