from pathlib import Path
from api.core.config import settings


class ModelNotLoadedError(Exception):

    def __init__(self, internal_message: str, client_message: str):

        self.internal_message = internal_message
        self.client_message = client_message

        super().__init__(self.internal_message)


class PreprocessingError(Exception):

    def __init__(self, internal_message: str, client_message: str):

        self.internal_message = internal_message
        self.client_message = client_message

        super().__init__(self.internal_message)


class PredictionError(Exception):

    def __init__(self, internal_message: str, client_message: str):

        self.internal_message = internal_message
        self.client_message = client_message

        super().__init__(self.internal_message)


class DriftReportError(Exception):

    def __init__(self, internal_message: str, client_message: str):

        self.internal_message = internal_message
        self.client_message = client_message

        super().__init__(self.internal_message)
