# app/api/v1/endpoints/weather.py
from fastapi import APIRouter, HTTPException
from app.services.weather_service import WeatherService

router = APIRouter()

@router.get("/reverse-geocode")
async def reverse_geocode(lat: float, lon: float):
    """
    Convert latitude & longitude into address metadata.
    """
    try:
        return await WeatherService.reverse_geocode(lat, lon)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
