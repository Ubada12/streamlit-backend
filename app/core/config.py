# app/core/config.py
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    """
    Central configuration using Pydantic BaseSettings.
    Loads environment variables from `.env` automatically.
    Also contains static ML-related configurations.
    """

    # -------------------------
    # Project Info
    # -------------------------
    PROJECT_NAME: str = "Flood & Weather API"
    VERSION: str = "1.0.0"

    # -------------------------
    # Security & API Keys
    # -------------------------
    API_KEY: str  # Your general API key
    AWS_ACCESS_KEY: str  # Required for AWS S3
    AWS_SECRET_KEY: str
    BUCKET_NAME: str
    MSG91_API_URL: str = "https://control.msg91.com/api/v5/email/send"  # Default MSG91 endpoint
    MSG91_AUTH_KEY: str  # Required for sending emails

    # -------------------------
    # CORS
    # -------------------------
    ALLOWED_ORIGINS: List[str] = ["*"]

    # -------------------------
    # ML Configurations
    # -------------------------
    INPUT_COLUMNS: List[str] = [
        "app_temp", "clouds", "dewpt", "dhi", "dni", "elev_angle",
        "ghi", "pres", "rh", "slp", "solar_rad", "temp", "uv", "vis",
        "wind_dir", "wind_spd", "hour", "month"
    ]

    OUTPUT_COLUMNS: List[str] = ["weather", "precip"]

    META_DATA: List[str] = ['timezone', 'temp', 'sources', 'country_code', 'city_name']

    # -------------------------
    # Model Paths
    # -------------------------
    VGG16_MODEL_PATH: str = "ml_models/vgg16_model.keras"
    XGB_MODEL_PATH: str = "ml_models/xgb.pkl"
    SCALER_PATH: str = "ml_models/scaler.pkl"

    class Config:
        env_file = ".env"  # Auto-load environment variables from this file


# Instantiate settings (singleton style)
settings = Settings()
