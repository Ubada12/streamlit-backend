import pandas as pd
import requests

from app.core.config import settings


class DataProcessing:
    def __init__(self, lat, lon):
        self.lat = lat
        self.lon = lon

    def fetch_data(self):
        """Fetch weather data from API."""
        try:
            uri = (
                f"https://api.weatherbit.io/v2.0/current"
                f"?lat={self.lat}&lon={self.lon}&key={settings.API_KEY}"
            )
            response = requests.get(uri, timeout=15)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Fetching data error: {e}") from e

    def process_data(self):
        """Process fetched weather data into a structured format."""
        data = self.fetch_data()
        if "data" not in data:
            raise ValueError("Invalid response format: Missing 'data' key")

        df = pd.DataFrame(data["data"])

        if "ts" in df.columns:
            df["timestamp"] = pd.to_datetime(df["ts"], unit="s")
            df["hour"] = df["timestamp"].dt.hour
            df["month"] = df["timestamp"].dt.month
            df.drop(["ts", "timestamp"], axis=1, inplace=True)

        if "weather" in df.columns:
            df["weather"] = df["weather"].apply(
                lambda x: x["description"] if isinstance(x, dict) else x
            )

        inputs = df[settings.INPUT_COLUMNS]
        outputs = df[settings.OUTPUT_COLUMNS]
        metadata = df[settings.META_DATA]

        return {
            "inputs": inputs.to_dict(orient="records"),
            "outputs": outputs.to_dict(orient="records"),
            "metadata": metadata.to_dict(orient="records"),
        }
