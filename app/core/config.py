# app/core/config.py
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """
    Central configuration using Pydantic BaseSettings.
    Loads environment variables from `.env` automatically.
    """

    # ── Project info ───────────────────────────────────────────────
    PROJECT_NAME: str = "Flood & Weather API"
    VERSION:      str = "1.0.0"

    # ── API keys ───────────────────────────────────────────────────
    API_KEY:         str   # Weatherbit API key
    AWS_ACCESS_KEY:  str   # AWS S3 access key
    AWS_SECRET_KEY:  str   # AWS S3 secret key
    BUCKET_NAME:     str   # S3 bucket name

    # ── MSG91 email ────────────────────────────────────────────────
    MSG91_API_URL:    str = "https://control.msg91.com/api/v5/email/send"
    MSG91_AUTH_KEY:   str   # MSG91 auth key
    MSG91_TEMPLATE_ID: str  # MSG91 email template ID for flood alerts
    SENDER_EMAIL:     str   # From-address registered in MSG91
    SENDER_NAME:      str = "XAI-FLOWS Alert System"
    RECIPIENT_EMAIL:  str   # Destination address (e.g. BMC flood control)
    RECIPIENT_NAME:   str = "BMC Flood Control Department"

    # ── CORS ───────────────────────────────────────────────────────
    ALLOWED_ORIGINS: List[str] = ["*"]

    # ── ML: feature columns ────────────────────────────────────────
    INPUT_COLUMNS: List[str] = [
        "app_temp", "clouds", "dewpt", "dhi", "dni", "elev_angle",
        "ghi", "pres", "rh", "slp", "solar_rad", "temp", "uv", "vis",
        "wind_dir", "wind_spd", "hour", "month",
    ]
    OUTPUT_COLUMNS: List[str] = ["weather", "precip"]
    META_DATA:      List[str] = ["timezone", "temp", "sources", "country_code", "city_name"]

    # ── ML: model paths ────────────────────────────────────────────
    VGG16_MODEL_PATH: str = "ml_models/vgg16_model.keras"
    XGB_MODEL_PATH:   str = "ml_models/xgb.pkl"
    SCALER_PATH:      str = "ml_models/scaler.pkl"

    class Config:
        env_file = ".env"


settings = Settings()
