from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from api.services.model_loader import initialize_artifacts
from api.services.prediction_logger import initialize_log_db
from api.core.config import settings
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from api.routes.health import router as health_router
from api.routes.metrics import router as metrics_router
from api.routes.predict import router as predict_router
from api.core.errors import (
    ModelNotLoadedError,
    PreprocessingError,
    PredictionError,
    DriftReportError,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Load the model and the processor...")
    initialize_artifacts(settings.model_path, settings.preprocessor_path)
    initialize_log_db(settings.prediction_db_path)
    if Path(settings.reference_distributions_path).exists():
        yield
    print("Loading finished.")


app = FastAPI(
    title="ChurnOps",
    description="Customer churn prediction API with drift monitoring",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_methods=settings.allowed_methods,
)

app.include_router(health_router, prefix="/health")
app.include_router(metrics_router, prefix="/metrics")
app.include_router(predict_router, prefix="/predict")


@app.exception_handler(ModelNotLoadedError)
async def model_not_loaded_handler(request: Request, exec: ModelNotLoadedError):
    return HTTPException(status_code=503, detail=exec.client_message)


@app.exception_handler(PreprocessingError)
async def preprocessing_handler(request: Request, exec: PreprocessingError):
    return HTTPException(status_code=422, detail=exec.client_message)


@app.exception_handler(PredictionError)
async def prediction_handler(request: Request, exec: PredictionError):
    return HTTPException(status_code=500, detail=exec.client_message)


@app.exception_handler(DriftReportError)
async def drift_report_handler(request: Request, exec: DriftReportError):
    return HTTPException(status_code=500, detail=exec.client_message)
