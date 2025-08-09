import pandas as pd
from app.utils.model import FloodPredictor
from app.utils.mlpipeline import MlPipeline
from app.utils.weather import DataProcessing


class HeuristicModel:
    def __init__(self, image_path, lon, lat):
        self.image_path = image_path
        self.lon = lon
        self.lat = lat

        # Get weather data
        data = DataProcessing(lat, lon).process_data()
        self.input_data = data.get("inputs")[0]
        self.output_data = data.get("outputs")[0]
        self.metadata = data.get("metadata")[0]
        # SHAP values for weather data
        self.weather_shap_value = MlPipeline(pd.DataFrame([self.input_data])).explain_shap()

        # Predict blockage in drain using CNN model
        flood_json = FloodPredictor().predict(image_path=image_path)
        self.blockage = flood_json.get("blockage")  # 0 = No blockage, 1 = Blockage
        self.blockage_prob = flood_json.get("probability")
        self.blockage_shap_value = flood_json.get("shap_values")

    def predict(self):
        precip = self.output_data['precip']
        weather = self.output_data["weather"]
        rh = self.output_data.get("rh", 0)  # Humidity (%), fallback = 0 if missing
        blockage = self.blockage              # 0 = full, 1 = no blockage, 2 = partial
        blockage_prob = self.blockage_prob    # Confidence: float between 0 and 1

        # Define conditions
        high_precip = precip > 10
        moderate_precip = 5 < precip <= 10
        light_precip = 0 < precip <= 5
        no_precip = precip == 0

        bad_weather = weather in ["Heavy rain", "Moderate rain", "Light rain", "Overcast clouds"]
        mild_weather = weather in ["Drizzle", "Scattered clouds", "Haze"]
        foggy_weather = weather in ["Fog", "Mist"]
        dry_weather = weather in ["Clear", "Sunny", "Few clouds"]

        # Define confidence levels
        very_confident = blockage_prob > 0.85
        confident = 0.7 < blockage_prob <= 0.85
        uncertain = 0.5 < blockage_prob <= 0.7
        very_uncertain = blockage_prob <= 0.5

        # Edge case: Post-rain signal (0 precip but fog + high humidity)
        post_rain_suspected = (
            blockage == 2 and
            0.6 < blockage_prob < 0.75 and
            no_precip and
            foggy_weather and
            rh > 85
        )
        if post_rain_suspected:
            return {
                "flood_risk": "Moderate",
                "reason": "Partial blockage with post-rain fog and high humidity — signs of recent flooding."
            }

        # 🌊 FULL BLOCKAGE (blockage == 0)
        if blockage == 0:
            if very_confident or confident:
                if high_precip and bad_weather:
                    return {"flood_risk": "High", "reason": "Severe blockage with heavy rain and bad weather."}
                elif high_precip or bad_weather:
                    return {"flood_risk": "High", "reason": "Severe blockage with either heavy rain or bad weather."}
                else:
                    return {"flood_risk": "High", "reason": "Severe blockage alone can trigger urban flooding."}

            elif uncertain:
                if bad_weather or moderate_precip:
                    return {"flood_risk": "Moderate", "reason": "Possible full blockage with weather threats. Confidence is low."}
                else:
                    return {"flood_risk": "Moderate", "reason": "Suspected full blockage. Weather conditions are neutral."}
    
            elif very_uncertain:
                if weather in ["Haze", "Fog", "Mist"]:
                    return {"flood_risk": "Moderate", "reason": "Low-confidence full blockage with hazy visibility."}
                else:
                    return {"flood_risk": "Low", "reason": "Uncertain full blockage but no strong weather signals."}

        # ⚠️ PARTIAL BLOCKAGE (blockage == 2)
        elif blockage == 2:
            if very_confident or confident:
                if high_precip and bad_weather:
                    return {"flood_risk": "Moderate", "reason": "Partial blockage with heavy rain and bad weather."}
                elif moderate_precip and mild_weather:
                    return {"flood_risk": "Moderate", "reason": "Partial blockage with moderate rain and mild weather."}
                elif foggy_weather and rh > 85:
                    return {"flood_risk": "Moderate", "reason": "Partial blockage with fog and high humidity — signs of residual flooding."}
                elif mild_weather or light_precip:
                    return {"flood_risk": "Moderate", "reason": "Partial blockage with mild conditions."}
                elif dry_weather and no_precip:
                    return {"flood_risk": "Moderate", "reason": "Partial blockage but dry and clear weather."}
                elif foggy_weather:
                    return {"flood_risk": "Moderate", "reason": "Partial blockage with foggy weather need to be alert."}
                else:
                    return {"flood_risk": "Low", "reason": "Partial blockage with unknown weather pattern."}
            else:
                if foggy_weather and rh > 85:
                    return {"flood_risk": "Moderate", "reason": "Suspected partial blockage with fog and high humidity."}
                else:
                    return {"flood_risk": "Low", "reason": "Suspected partial blockage, but model is not confident."}

        # ✅ NO BLOCKAGE (blockage == 1)
        elif blockage == 1:
            if high_precip and bad_weather:
                return {"flood_risk": "Low", "reason": "No blockage, but heavy rain and bad weather could cause water accumulation."}
            elif moderate_precip and bad_weather:
                return {"flood_risk": "Low", "reason": "No blockage, but mild flooding may occur due to weather."}
            else:
                return {"flood_risk": "Minimal", "reason": "No blockage and no significant environmental threat."}

        # 🛑 Fallback for unexpected scenarios
        return {
            "flood_risk": "Minimal",
            "reason": "Default fallback: conditions unclear, no strong flood indicators.",
            "debug": {
                "blockage": blockage,
                "blockage_prob": blockage_prob,
                "precip": precip,
                "weather": weather
            }
        }


