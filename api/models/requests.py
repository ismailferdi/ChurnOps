from core.features import FEATURE_ORDER
from pydantic import BaseModel, ConfigDict, Field
import pandas as pd
from typing import Literal


class CustomerFeatures(BaseModel):

    model_config = ConfigDict(str_strip_whitespace=True)

    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(ge=0, le=100)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["No phone service", "No", "Yes"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["No", "Yes", "No internet service"]
    OnlineBackup: Literal["No", "Yes", "No internet service"]
    DeviceProtection: Literal["No", "Yes", "No internet service"]
    TechSupport: Literal["No", "Yes", "No internet service"]
    StreamingTV: Literal["No", "Yes", "No internet service"]
    StreamingMovies: Literal["No", "Yes", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    MonthlyCharges: float = Field(ge=0.0)
    TotalCharges: float = Field(ge=0.0)

    def to_dataframe(self) -> pd.DataFrame:

        return pd.DataFrame(
            [
                [
                    self.gender,
                    self.SeniorCitizen,
                    self.Partner,
                    self.Dependents,
                    self.tenure,
                    self.PhoneService,
                    self.MultipleLines,
                    self.InternetService,
                    self.OnlineSecurity,
                    self.OnlineBackup,
                    self.DeviceProtection,
                    self.TechSupport,
                    self.StreamingTV,
                    self.StreamingMovies,
                    self.Contract,
                    self.PaperlessBilling,
                    self.PaymentMethod,
                    self.MonthlyCharges,
                    self.TotalCharges,
                ]
            ],
            columns=FEATURE_ORDER,
        )
