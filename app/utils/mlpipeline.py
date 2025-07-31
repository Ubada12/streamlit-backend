import pickle
import shap
import numpy as np
import pandas as pd
import logging
from app.core.config import settings

logging.basicConfig(level=logging.INFO)

class MlPipeline:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize ML Pipeline with preloaded model and scaler.
        """
        self.data = data

        # Load model and scaler once at initialization
        self.scaler = self.load_scaler()
        self.model = self.load_model()

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
        elif isinstance(self.data, pd.DataFrame):
            return self.data.values  # Convert DataFrame to NumPy array
        else:
            raise TypeError("Input data must be a NumPy array or Pandas DataFrame.")

    def scale(self):
        """Scale the input data using the preloaded scaler."""
        validated_data = self.validate_data()
        try:
            scaled_data = self.scaler.transform(validated_data)
            return scaled_data
        except Exception as e:
            logging.error(f"Error in scaling data: {e}")
            return None

    def explain_shap(self):
        """Compute SHAP values for the model's predictions."""
        try:
            explainer = shap.Explainer(self.model, feature_names=self.data.columns)
            shap_values = explainer(self.scale())
            print(type(shap_values))
            return shap_values
        except Exception as e:
            logging.error(f"Error computing SHAP values: {e}")
            return None

    def predict(self):
        """Generate predictions using the pre-trained model."""
        try:
            scaled_data = self.scale()
            if scaled_data is not None:
                prediction = self.model.predict(scaled_data)
                return prediction
        except Exception as e:
            logging.error(f"Prediction error: {e}")
            return None
