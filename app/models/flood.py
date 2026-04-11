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
    weather_data: Optional[Any]
    weather_prediction: Optional[Any]
    weather_metadata: Optional[Any]
    weather_shap_value: Optional[Any]
    drain_blockage: Optional[Any]
    drain_blockage_prob: Optional[Any]
    drain_blockage_shape_value: Optional[Any]
