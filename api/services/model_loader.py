from api.core.errors import ModelNotLoadedError
import joblib

_model = None
_preprocessor = None


def load_model(path: str):

    try:
        return joblib.load(path)
    except FileNotFoundError:
        raise ModelNotLoadedError(
            internal_message=f"Model file not found: {path}",
            client_message="Model is unavailable.",
        )


def load_preprocessor(path: str):

    try:
        return joblib.load(path)
    except FileNotFoundError:
        raise ModelNotLoadedError(
            internal_message=f"Preprocessor file not found: {path}",
            client_message="Preprocessor is unavailable.",
        )


def get_model():
    if _model is not None:
        return _model

    raise ModelNotLoadedError(
        internal_message="Model is not loaded", client_message="Model is unavailable."
    )


def get_preprocessor():
    if _preprocessor is not None:
        return _preprocessor

    raise ModelNotLoadedError(
        internal_message=f"Preprocessor file not found",
        client_message="Preprocessor is unavailable.",
    )


def initialize_artifacts(model_path: str, preprocessor_path: str):
    global _model, _preprocessor

    _model = load_model(model_path)
    _preprocessor = load_preprocessor(preprocessor_path)
