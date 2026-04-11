import pandas as pd

from app.utils.model import FloodPredictor
from app.utils.mlpipeline import MlPipeline
from app.utils.weather import DataProcessing


class HeuristicModel:
    def __init__(self, image_path, lon, lat, vgg_model=None, xgb_model=None, scaler=None):
        self.image_path = image_path
        self.lon = lon
        self.lat = lat

        data = DataProcessing(lat, lon).process_data()
        self.input_data = data.get("inputs")[0]
        self.output_data = data.get("outputs")[0]
        self.metadata = data.get("metadata")[0]

        self.weather_shap_value = MlPipeline(
            pd.DataFrame([self.input_data]),
            model=xgb_model,
            scaler=scaler,
        ).explain_shap()

        flood_json = FloodPredictor(model=vgg_model).predict(image_path=image_path)
        if "error" in flood_json:
            raise RuntimeError(f"Drain blockage prediction failed: {flood_json['error']}")

        self.blockage = flood_json.get("blockage")  # 0 = full, 1 = no blockage, 2 = partial
        self.blockage_prob = flood_json.get("probability")
        self.blockage_shap_value = flood_json.get("shap_values")

    def predict(self):
        precip = self.output_data["precip"]
        weather = self.output_data["weather"]
        rh = self.input_data.get("rh", 0)
        blockage = self.blockage
        blockage_prob = self.blockage_prob

        high_precip = precip > 10
        moderate_precip = 5 < precip <= 10
        light_precip = 0 < precip <= 5
        no_precip = precip == 0

        bad_weather = weather in ["Heavy rain", "Moderate rain", "Light rain", "Overcast clouds"]
        mild_weather = weather in ["Drizzle", "Scattered clouds", "Haze"]
        foggy_weather = weather in ["Fog", "Mist"]
        dry_weather = weather in ["Clear", "Sunny", "Few clouds"]

        very_confident = blockage_prob > 0.85
        confident = 0.7 < blockage_prob <= 0.85
        uncertain = 0.5 < blockage_prob <= 0.7
        very_uncertain = blockage_prob <= 0.5

        post_rain_suspected = (
            blockage == 2
            and 0.6 < blockage_prob < 0.75
            and no_precip
            and foggy_weather
            and rh > 85
        )
        if post_rain_suspected:
            return {
                "flood_risk": "Moderate",
                "reason": "Partial blockage with post-rain fog and high humidity signs of recent flooding.",
            }

        if blockage == 0:
            if very_confident or confident:
                if high_precip and bad_weather:
                    return {"flood_risk": "High", "reason": "Severe blockage with heavy rain and bad weather."}
                if high_precip or bad_weather:
                    return {"flood_risk": "High", "reason": "Severe blockage with either heavy rain or bad weather."}
                return {"flood_risk": "High", "reason": "Severe blockage alone can trigger urban flooding."}

            if uncertain:
                if bad_weather or moderate_precip:
                    return {"flood_risk": "Moderate", "reason": "Possible full blockage with weather threats. Confidence is low."}
                return {"flood_risk": "Moderate", "reason": "Suspected full blockage. Weather conditions are neutral."}

            if very_uncertain:
                if weather in ["Haze", "Fog", "Mist"]:
                    return {"flood_risk": "Moderate", "reason": "Low-confidence full blockage with hazy visibility."}
                return {"flood_risk": "Low", "reason": "Uncertain full blockage but no strong weather signals."}

        elif blockage == 2:
            if very_confident or confident:
                if high_precip and bad_weather:
                    return {"flood_risk": "Moderate", "reason": "Partial blockage with heavy rain and bad weather."}
                if moderate_precip and mild_weather:
                    return {"flood_risk": "Moderate", "reason": "Partial blockage with moderate rain and mild weather."}
                if foggy_weather and rh > 85:
                    return {"flood_risk": "Moderate", "reason": "Partial blockage with fog and high humidity signs of residual flooding."}
                if mild_weather or light_precip:
                    return {"flood_risk": "Moderate", "reason": "Partial blockage with mild conditions."}
                if dry_weather and no_precip:
                    return {"flood_risk": "Moderate", "reason": "Partial blockage but dry and clear weather."}
                if foggy_weather:
                    return {"flood_risk": "Moderate", "reason": "Partial blockage with foggy weather need to be alert."}
                return {"flood_risk": "Low", "reason": "Partial blockage with unknown weather pattern."}

            if foggy_weather and rh > 85:
                return {"flood_risk": "Moderate", "reason": "Suspected partial blockage with fog and high humidity."}
            return {"flood_risk": "Low", "reason": "Suspected partial blockage, but model is not confident."}

        elif blockage == 1:
            if high_precip and bad_weather:
                return {"flood_risk": "Low", "reason": "No blockage, but heavy rain and bad weather could cause water accumulation."}
            if moderate_precip and bad_weather:
                return {"flood_risk": "Low", "reason": "No blockage, but mild flooding may occur due to weather."}
            return {"flood_risk": "Minimal", "reason": "No blockage and no significant environmental threat."}

        return {
            "flood_risk": "Minimal",
            "reason": "Default fallback: conditions unclear, no strong flood indicators.",
            "debug": {
                "blockage": blockage,
                "blockage_prob": blockage_prob,
                "precip": precip,
                "weather": weather,
            },
        }
