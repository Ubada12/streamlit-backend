# app/models/flood.py
from pydantic import BaseModel, Field
from typing import Dict, Optional, Any


class FloodPredictionRequest(BaseModel):
    lon: float = Field(..., example=72.8777, description="Longitude")
    lat: float = Field(..., example=19.0760, description="Latitude")


class FloodPredictionResponse(BaseModel):
    image: str
    longitude: float
    latitude: float
    prediction: Dict[str, str]
    weather_data: Optional[Any] = None
    weather_prediction: Optional[Any] = None
    weather_metadata: Optional[Any] = None
    weather_shap_value: Optional[Any] = None
    drain_blockage: Optional[Any] = None
    drain_blockage_prob: Optional[Any] = None
    drain_blockage_shap_value: Optional[Any] = (
        None  # was: drain_blockage_shape_value (typo)
    )
