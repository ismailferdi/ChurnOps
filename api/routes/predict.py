from fastapi import APIRouter, HTTPException
from api.models.requests import CustomerFeatures
from api.services.model_loader import get_model, get_preprocessor
from api.services.predictor import predict
from api.services.prediction_logger import log_prediction
from api.core.config import settings
from api.models.responses import PredictionResponse
from api.core.errors import PredictionError, PreprocessingError

router = APIRouter()


@router.post("/", response_model=PredictionResponse)
def post_predict(customer: CustomerFeatures) -> PredictionResponse:
    model = get_model()
    preprocessor = get_preprocessor()
    try:
        result = predict(customer, model, preprocessor)
        log_prediction(
            db_path=settings.prediction_db_path, customer=customer, result=result
        )
        return PredictionResponse(**result)
    except PreprocessingError as e:
        raise HTTPException(status_code=422, detail=e.client_message)
    except PredictionError as e:
        raise HTTPException(status_code=500, detail=e.client_message)
