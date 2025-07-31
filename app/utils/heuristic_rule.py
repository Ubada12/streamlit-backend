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
        blockage = self.blockage
        blockage_prob = self.blockage_prob

        # Define conditions for flooding
        high_precip = precip > 10  # Threshold for heavy rain
        bad_weather = weather in ["Heavy rain", "Moderate rain", "Light rain", "Overcast clouds"]
        severe_blockage = blockage == 0 and blockage_prob > 0.7

        # Heuristic rule
        if high_precip and bad_weather and severe_blockage:
            return {"flood_risk": "High", "reason": "Heavy rain, bad weather, and severe blockage detected."}
        elif (high_precip and bad_weather) or severe_blockage:
            return {"flood_risk": "Moderate", "reason": "Either heavy rain and bad weather or severe blockage detected."}
        elif precip > 5 and weather in ["Drizzle", "Scattered clouds"] and blockage == 1:
            return {"flood_risk": "Low", "reason": "Moderate precipitation and mild weather with blockage."}
        else:
            return {"flood_risk": "Minimal", "reason": "No significant blockage, precipitation, or bad weather detected."}
