from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

ENV_PATH = Path(__file__).parent.parent.parent / ".env"


class Settings(BaseSettings):

    model_config = SettingsConfigDict(env_file=ENV_PATH, env_file_encoding="utf-8")

    model_path: str
    preprocessor_path: str
    reference_distributions_path: str
    prediction_db_path: str
    mlflow_tracking_uri: str
    cors_allowed_origins: list[str]


settings = Settings()
