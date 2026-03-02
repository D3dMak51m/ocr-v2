import os
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """
    API_TOKEN: str = "your_secret_token"
    AIRFLOW_BASE_URL: str = "http://airflow-webserver:8080"
    AIRFLOW_USER: str = "admin"
    AIRFLOW_PASSWORD: str = "admin"
    TIKA_SERVER_URL: str = "http://tika-server:9998"
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(
        env_file = ".env",  # Allows loading from a .env file
        env_file_encoding = "utf-8"
    )

# Create a single instance to be used across the application
settings = Settings()
