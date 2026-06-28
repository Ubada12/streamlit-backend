# app/models/flood.py
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any


class FloodPredictionRequest(BaseModel):
    lon: float = Field(..., example=72.8777, description="Longitude")
    lat: float = Field(..., example=19.0760, description="Latitude")


class LocationInfo(BaseModel):
    """Reverse-geocoded location returned with every prediction."""
    latitude:  float
    longitude: float
    address:   str    # full Nominatim display_name
    city:      str    # locality / city name


class WeatherInfo(BaseModel):
    """
    Typed, flat weather object built from Weatherbit API response.
    precipitation is taken from output_data (live observed value),
    NOT from input_data where it is always 0.
    """
    temp:          float   # °C  — current temperature
    app_temp:      float   # °C  — apparent / feels-like temperature
    humidity:      float   # %   — relative humidity  (input_data["rh"])
    wind_speed:    float   # m/s (input_data["wind_spd"])
    wind_dir:      float   # degrees
    uv:            float   # UV index
    pressure:      float   # hPa (input_data["pres"])
    visibility:    float   # km  (input_data["vis"])
    precipitation: float   # mm/hr — FROM output_data["precip"]  ← bug fix
    condition:     str     # e.g. "Heavy rain" — FROM output_data["weather"]
    clouds:        float   # % cloud coverage
    dewpt:         float   # °C dew point


class ShapPoint(BaseModel):
    feature: str
    value:   float


class FloodPredictionResponse(BaseModel):
    """
    Unified response returned by POST /predict-flood/.

    Changes from old shape:
    - location   : typed LocationInfo (was missing — frontend geocoded separately)
    - weather    : typed WeatherInfo  (was a raw Any blob)
    - alert_sent : bool               (email dispatch now happens server-side)
    - removed    : image, weather_data, weather_prediction, weather_metadata raw fields
    """
    prediction:               Dict[str, Any]
    location:                 LocationInfo
    weather:                  WeatherInfo
    weather_shap_value:       Optional[List[ShapPoint]]  = None
    drain_blockage:           Optional[int]               = None  # 0=full 1=none 2=partial
    drain_blockage_prob:      Optional[float]             = None
    drain_blockage_shap_value: Optional[Any]              = None
    alert_sent:               bool                        = False
