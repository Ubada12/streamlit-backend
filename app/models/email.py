# app/models/email.py
from pydantic import BaseModel, Field


class FloodAlertEmailPayload(BaseModel):
    """
    Strictly validated payload for a XAI-FLOWS flood alert email.
    All numeric weather values must be pre-formatted as strings with units
    by the caller (e.g. "32.4°C", "12.5 mm/hr") before sending to this endpoint.
    """

    # Event metadata
    Timestamp: str = Field(
        ..., description="ISO datetime of detection, e.g. '2025-06-25T14:32:00'"
    )
    risk: str = Field(
        ..., description="Flood risk level: High | Moderate | Low | Minimal"
    )

    # Location
    address: str = Field(
        ..., description="Full human-readable address from reverse geocoding"
    )
    latitude: str = Field(..., description="Latitude as string, e.g. '19.0760'")
    longitude: str = Field(..., description="Longitude as string, e.g. '72.8777'")

    # Weather — all values include units
    temperature: str = Field(..., description="e.g. '32.4°C'")
    precipitation: str = Field(..., description="e.g. '12.5 mm/hr'")
    wind_speed: str = Field(..., description="e.g. '5.2 m/s'")
    humidity: str = Field(..., description="e.g. '87%'")
    visibility: str = Field(..., description="e.g. '8.0 km'")
    condition: str = Field(..., description="e.g. 'Heavy rain'")
    uv_index: str = Field(..., description="e.g. '3'")
    pressure: str = Field(..., description="e.g. '1008.2 hPa'")
    cloud_coverage: str = Field(..., description="e.g. 'Overcast'")
