# app/services/flood_service.py
import os
import json
import tempfile
import pickle
import logging
from datetime import datetime, timezone

import httpx
from tensorflow.keras.models import load_model

from app.models.flood import (
    FloodPredictionRequest,
    FloodPredictionResponse,
    LocationInfo,
    WeatherInfo,
    ShapPoint,
)
from app.models.email import FloodAlertEmailPayload
from app.core.config import settings

logger = logging.getLogger(__name__)

# Risk levels that trigger an alert email
_ALERT_RISK_LEVELS = {"High", "Moderate"}

# Nominatim base URL
_NOMINATIM_URL = "https://nominatim.openstreetmap.org/reverse"


def _cloud_coverage_label(clouds: float) -> str:
    """Convert numeric cloud % to a readable label (mirrors frontend getCloudCoverage)."""
    if clouds >= 90:
        return "Overcast"
    if clouds >= 70:
        return "Mostly Cloudy"
    if clouds >= 40:
        return "Partly Cloudy"
    if clouds >= 10:
        return "Few Clouds"
    return "Clear Sky"


def _build_weather_info(input_data: dict, output_data: dict) -> WeatherInfo:
    """
    Build a clean typed WeatherInfo from the two Weatherbit data dicts.
    precipitation comes from output_data (live observed value) — NOT input_data
    where the field is always 0 because it is a model input feature, not output.
    """
    return WeatherInfo(
        temp=float(input_data.get("temp", 0)),
        app_temp=float(input_data.get("app_temp", 0)),
        humidity=float(input_data.get("rh", 0)),
        wind_speed=float(input_data.get("wind_spd", 0)),
        wind_dir=float(input_data.get("wind_dir", 0)),
        uv=float(input_data.get("uv", 0)),
        pressure=float(input_data.get("pres", 0)),
        visibility=float(input_data.get("vis", 0)),
        precipitation=float(output_data.get("precip", 0)),   # ← the fix
        condition=str(output_data.get("weather", "Unknown")),
        clouds=float(input_data.get("clouds", 0)),
        dewpt=float(input_data.get("dewpt", 0)),
    )


async def _reverse_geocode(lat: float, lon: float) -> LocationInfo:
    """
    Call Nominatim to resolve (lat, lon) → address.
    Falls back to a coordinate-based placeholder on any failure.
    Never raises — a missing address must not abort the prediction.
    """
    try:
        headers = {"User-Agent": "xai-flows-api/1.0"}
        url = f"{_NOMINATIM_URL}?lat={lat}&lon={lon}&format=json"
        async with httpx.AsyncClient(timeout=8) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            address = data.get("display_name", "")
            city = data.get("address", {}).get("city", "")
            if address:
                return LocationInfo(
                    latitude=lat, longitude=lon,
                    address=address,
                    city=city or address.split(",")[0].strip(),
                )
    except Exception as e:
        logger.warning(f"Reverse geocode failed (non-critical): {e}")

    # Fallback
    return LocationInfo(
        latitude=lat, longitude=lon,
        address=f"{lat:.4f}°N, {lon:.4f}°E",
        city="Unknown",
    )


async def _send_alert_email(
    prediction_result: dict,
    weather: WeatherInfo,
    location: LocationInfo,
) -> bool:
    """
    Build FloodAlertEmailPayload and dispatch via EmailService.
    Returns True if the email was sent successfully, False otherwise.
    Never raises — a failed email must not abort the prediction response.
    """
    from app.services.email_service import EmailService   # local import avoids circular risk

    try:
        payload = FloodAlertEmailPayload(
            Timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
            risk=prediction_result["flood_risk"],
            address=location.address,
            latitude=f"{location.latitude:.6f}",
            longitude=f"{location.longitude:.6f}",
            temperature=f"{weather.temp:.1f}°C",
            precipitation=f"{weather.precipitation:.1f} mm/hr",
            wind_speed=f"{weather.wind_speed:.1f} m/s",
            humidity=f"{weather.humidity:.0f}%",
            visibility=f"{weather.visibility:.1f} km",
            condition=weather.condition,
            uv_index=str(int(weather.uv)),
            pressure=f"{weather.pressure:.1f} hPa",
            cloud_coverage=_cloud_coverage_label(weather.clouds),
        )
        await EmailService.send_flood_alert(payload)
        logger.info(
            f"Flood alert email sent — risk={payload.risk}, address={payload.address}"
        )
        return True
    except Exception as e:
        logger.error(f"Flood alert email failed (non-critical): {e}")
        return False


class FloodModelService:
    # Class-level cache — loaded once at startup, reused for every request
    vgg_model = None
    xgb_model = None
    scaler    = None

    @classmethod
    def load_models(cls):
        """Load all ML models once at startup. Idempotent."""
        if cls.vgg_model is None:
            try:
                cls.vgg_model = load_model(settings.VGG16_MODEL_PATH)
                logger.info("VGG16 model loaded.")
            except Exception as e:
                logger.error(f"Error loading VGG16: {e}")

        if cls.xgb_model is None:
            try:
                with open(settings.XGB_MODEL_PATH, "rb") as f:
                    cls.xgb_model = pickle.load(f)
                logger.info("XGBoost model loaded.")
            except Exception as e:
                logger.error(f"Error loading XGBoost: {e}")

        if cls.scaler is None:
            try:
                with open(settings.SCALER_PATH, "rb") as f:
                    cls.scaler = pickle.load(f)
                logger.info("Scaler loaded.")
            except Exception as e:
                logger.error(f"Error loading Scaler: {e}")

    @staticmethod
    async def predict_flood(image_file, request_json: str) -> FloodPredictionResponse:
        """
        Unified flood prediction pipeline.

        Runs synchronous ML work (VGG16 + XGBoost + SHAP), then in parallel
        dispatches reverse geocoding and — on High/Moderate risk — an alert
        email. All network calls are non-blocking async; failures are logged
        but never surface as prediction errors.
        """
        # Deferred import — breaks the heuristic_rule → flood_service circular chain
        from app.utils.heuristic_rule import HeuristicModel

        try:
            request_data  = json.loads(request_json)
            request_model = FloodPredictionRequest(**request_data)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Invalid request payload: {e}")

        if not (FloodModelService.vgg_model and FloodModelService.xgb_model and FloodModelService.scaler):
            raise RuntimeError(
                "Models not loaded. Ensure FloodModelService.load_models() ran at startup."
            )

        suffix = os.path.splitext(image_file.filename or "upload")[1] or ".jpg"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        try:
            tmp.write(image_file.file.read())
            tmp.flush()
            tmp.close()

            # ── 1. Run synchronous ML pipeline ──────────────────────────────
            heuristic_model = HeuristicModel(
                image_path=tmp.name,
                lon=request_model.lon,
                lat=request_model.lat,
                vgg_model=FloodModelService.vgg_model,
                xgb_model=FloodModelService.xgb_model,
                scaler=FloodModelService.scaler,
            )
            prediction_result = heuristic_model.predict()

            # ── 2. Build clean weather object ────────────────────────────────
            weather = _build_weather_info(
                heuristic_model.input_data,
                heuristic_model.output_data,
            )

            # ── 3. Build SHAP data ───────────────────────────────────────────
            weather_shap: list[ShapPoint] | None = None
            if (
                hasattr(heuristic_model, "weather_shap_value")
                and hasattr(heuristic_model.weather_shap_value, "values")
            ):
                weather_shap = [
                    ShapPoint(feature=f, value=v)
                    for f, v in zip(
                        heuristic_model.weather_shap_value.feature_names,
                        heuristic_model.weather_shap_value.values.tolist()[0],
                    )
                ]

            drain_shap = (
                heuristic_model.blockage_shap_value.values.tolist()
                if hasattr(heuristic_model, "blockage_shap_value")
                and hasattr(heuristic_model.blockage_shap_value, "values")
                else None
            )

            # ── 4. Reverse geocode (async, graceful fallback) ────────────────
            location = await _reverse_geocode(request_model.lat, request_model.lon)

            # ── 5. Send alert email if risk warrants it ───────────────────────
            alert_sent = False
            if prediction_result.get("flood_risk") in _ALERT_RISK_LEVELS:
                alert_sent = await _send_alert_email(prediction_result, weather, location)

            return FloodPredictionResponse(
                prediction=prediction_result,
                location=location,
                weather=weather,
                weather_shap_value=weather_shap,
                drain_blockage=int(heuristic_model.blockage),
                drain_blockage_prob=float(heuristic_model.blockage_prob),
                drain_blockage_shap_value=drain_shap,
                alert_sent=alert_sent,
            )

        except Exception as e:
            raise RuntimeError(f"Flood prediction failed: {e}")
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
