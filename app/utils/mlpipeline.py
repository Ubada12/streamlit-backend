import logging
import shap
import numpy as np
import pandas as pd
from app.services.flood_service import FloodModelService

logger = logging.getLogger(__name__)


class MlPipeline:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize ML Pipeline using the cached models from FloodModelService.
        Models must have been loaded at startup via FloodModelService.load_models().
        """
        self.data = data
        self.scaler = FloodModelService.scaler
        self.model = FloodModelService.xgb_model

        if self.scaler is None or self.model is None:
            raise ValueError(
                "Model or scaler not loaded. Ensure FloodModelService.load_models() "
                "was called at startup."
            )

    def validate_data(self):
        """Ensure input data is a DataFrame or NumPy array before scaling."""
        if isinstance(self.data, np.ndarray):
            return self.data
        elif isinstance(self.data, pd.DataFrame):
            return self.data.values
        else:
            raise TypeError("Input data must be a NumPy array or Pandas DataFrame.")

    def scale(self):
        """Scale the input data using the preloaded scaler."""
        validated_data = self.validate_data()
        try:
            return self.scaler.transform(validated_data)
        except Exception as e:
            raise RuntimeError(f"Error scaling data: {e}")

    def explain_shap(self):
        """Compute SHAP values for the XGBoost model's predictions."""
        try:
            explainer = shap.Explainer(self.model, feature_names=self.data.columns.tolist())
            shap_values = explainer(self.scale())
            return shap_values
        except Exception as e:
            logger.error(f"Error computing SHAP values: {e}")
            return None

    def predict(self):
        """Generate predictions using the pre-trained XGBoost model."""
        try:
            return self.model.predict(self.scale())
        except Exception as e:
            raise RuntimeError(f"Prediction error: {e}")
