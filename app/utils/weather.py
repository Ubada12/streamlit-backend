import pandas as pd
import numpy as np
import requests
from app.core.config import settings

from dotenv import load_dotenv
import os

load_dotenv("/home/nooman/Documents/flood_predicter/.env")
class DataProcessing:
    def __init__(self, lat, lon):
        self.lat = lat
        self.lon = lon  # Fixing incorrect assignment

    def fetch_data(self):
        """Fetch weather data from API"""
        try:
            uri = f"https://api.weatherbit.io/v2.0/current?lat={self.lat}&lon={self.lon}&key=41065794439a472e920f72ff43512ffc"
            print(uri)
            response = requests.get(uri)
            response.raise_for_status()  # Raise an error for non-200 responses
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Fetching Data Error: {e}")

    def process_data(self):
        """Process fetched weather data into a structured format"""
        data = self.fetch_data()
        if "data" not in data:
            raise ValueError("Invalid response format: Missing 'data' key")

        df = pd.DataFrame(data["data"])

        # Convert 'ts' column to datetime
        if "ts" in df.columns:
            df["timestamp"] = pd.to_datetime(df["ts"], unit="s")
            df["hour"] = df["timestamp"].dt.hour
            df["month"] = df["timestamp"].dt.month
            df.drop(["ts", "timestamp"], axis=1, inplace=True)

        # Extract weather description
        if "weather" in df.columns:
            df["weather"] = df["weather"].apply(lambda x: x["description"] if isinstance(x, dict) else x)

        # Keep only selected columns
        inputs = df[settings.INPUT_COLUMNS]
        outputs = df[settings.OUTPUT_COLUMNS]
        metadata = df[settings.META_DATA]

        return {
              "inputs": inputs.to_dict(orient="records"),
              "outputs": outputs.to_dict(orient="records"),
              "metadata": metadata.to_dict(orient="records")
        }