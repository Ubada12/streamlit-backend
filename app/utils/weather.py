import logging
import pandas as pd
import requests
from app.core.config import settings

logger = logging.getLogger(__name__)


class DataProcessing:
    def __init__(self, lat, lon):
        self.lat = lat
        self.lon = lon

    def fetch_data(self):
        """Fetch weather data from Weatherbit API."""
        try:
            uri = f"https://api.weatherbit.io/v2.0/current?lat={self.lat}&lon={self.lon}&key={settings.API_KEY}"
            response = requests.get(uri, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Fetching Data Error: {e}")

    def process_data(self):
        """Process fetched weather data into a structured format."""
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

        # Extract weather description from nested dict
        if "weather" in df.columns:
            df["weather"] = df["weather"].apply(
                lambda x: x["description"] if isinstance(x, dict) else x
            )

        # Validate required columns exist
        missing_input = [c for c in settings.INPUT_COLUMNS if c not in df.columns]
        missing_output = [c for c in settings.OUTPUT_COLUMNS if c not in df.columns]
        missing_meta = [c for c in settings.META_DATA if c not in df.columns]

        if missing_input:
            raise ValueError(
                f"Missing input columns from API response: {missing_input}"
            )
        if missing_output:
            raise ValueError(
                f"Missing output columns from API response: {missing_output}"
            )
        if missing_meta:
            logger.warning(
                f"Missing metadata columns (will be skipped): {missing_meta}"
            )
            available_meta = [c for c in settings.META_DATA if c in df.columns]
        else:
            available_meta = settings.META_DATA

        inputs = df[settings.INPUT_COLUMNS]
        outputs = df[settings.OUTPUT_COLUMNS]
        metadata = df[available_meta]

        return {
            "inputs": inputs.to_dict(orient="records"),
            "outputs": outputs.to_dict(orient="records"),
            "metadata": metadata.to_dict(orient="records"),
        }
