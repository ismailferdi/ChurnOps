from fastapi import APIRouter
from api.services.model_loader import get_model, get_preprocessor
from api.core.errors import ModelNotLoadedError
from api.models.responses import HelthResponse

router = APIRouter()


@router.get("/", response_model=HelthResponse)
def health():
    model_loaded = False
    preprocessor_loaded = False

    try:
        model_loaded = get_model() is not None
    except:
        pass

    try:
        preprocessor_loaded = get_preprocessor() is not None
    except:
        pass

    return HelthResponse(
        status="ok", model_loaded=model_loaded, preprocessor_loaded=preprocessor_loaded
    )
