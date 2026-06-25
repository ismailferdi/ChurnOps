from fastapi import APIRouter, HTTPException
from monitoring.report import get_drift_report
from api.core.config import settings
from api.models.responses import MetricsResponse
from api.core.errors import DriftReportError

router = APIRouter()


@router.get("/", response_model=MetricsResponse)
def get_metrics(window: int = 500):
    try:
        drift_report = get_drift_report(settings.prediction_db_path, window)
        if drift_report["total_predictions"] == 0:
            return MetricsResponse(
                total_predictions=0, window_size=window, drift_metrics=[]
            )

        return MetricsResponse(window_size=window, **drift_report)
    except DriftReportError as e:
        return HTTPException(status_code=500, detail=e.client_message)
