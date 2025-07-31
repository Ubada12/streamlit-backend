# app/services/weather_service.py
import httpx

class WeatherService:
    BASE_URL = "https://nominatim.openstreetmap.org/reverse"

    @staticmethod
    async def reverse_geocode(lat: float, lon: float) -> dict:
        """
        Fetch address details from coordinates using Nominatim API.
        """
        if lat is None or lon is None:
            raise ValueError("Latitude and Longitude are required")

        headers = {"User-Agent": "flood-weather-api/1.0"}
        url = f"{WeatherService.BASE_URL}?lat={lat}&lon={lon}&format=json"

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(url, headers=headers)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"Nominatim API returned error {e.response.status_code}")
        except Exception as e:
            raise RuntimeError(f"Reverse geocoding failed: {str(e)}")
