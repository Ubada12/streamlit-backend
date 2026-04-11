import logging
import pickle
from typing import Optional

import numpy as np
import pandas as pd
import shap

from app.core.config import settings

logging.basicConfig(level=logging.INFO)


class MlPipeline:
    def __init__(self, data: pd.DataFrame, model: Optional[object] = None, scaler: Optional[object] = None):
        """
        Initialize ML Pipeline with a preloaded model and scaler when available.
        """
        self.data = data
        self.scaler = scaler or self.load_scaler()
        self.model = model or self.load_model()

        if self.scaler is None or self.model is None:
            raise ValueError("Failed to load model or scaler. Check file paths!")

    def load_scaler(self):
        """Load the pre-trained scaler from a pickle file."""
        try:
            with open(settings.SCALER_PATH, "rb") as f:
                scaler = pickle.load(f)
            logging.info("Scaler loaded successfully.")
            return scaler
        except Exception as e:
            logging.error(f"Error loading scaler: {e}")
            return None

    def load_model(self):
        """Load the pre-trained XGBoost model from a pickle file."""
        try:
            with open(settings.XGB_MODEL_PATH, "rb") as f:
                model = pickle.load(f)
            logging.info("Model loaded successfully.")
            return model
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            return None

    def validate_data(self):
        """Ensure input data is a DataFrame or NumPy array before scaling."""
        if isinstance(self.data, np.ndarray):
            return self.data
        if isinstance(self.data, pd.DataFrame):
            return self.data.values
        raise TypeError("Input data must be a NumPy array or Pandas DataFrame.")

    def scale(self):
        """Scale the input data using the preloaded scaler."""
        validated_data = self.validate_data()
        try:
            return self.scaler.transform(validated_data)
        except Exception as e:
            logging.error(f"Error in scaling data: {e}")
            return None

    def explain_shap(self):
        """Compute SHAP values for the model's predictions."""
        try:
            scaled_data = self.scale()
            if scaled_data is None:
                return None

            explainer = shap.Explainer(self.model, feature_names=self.data.columns)
            return explainer(scaled_data)
        except Exception as e:
            logging.error(f"Error computing SHAP values: {e}")
            return None

    def predict(self):
        """Generate predictions using the pre-trained model."""
        try:
            scaled_data = self.scale()
            if scaled_data is not None:
                return self.model.predict(scaled_data)
        except Exception as e:
            logging.error(f"Prediction error: {e}")
            return None
