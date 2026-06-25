import logging
import numpy as np
from PIL import Image
from app.core.config import settings

logger = logging.getLogger(__name__)


class FloodPredictor:
    def __init__(self):
        self.img_size = (256, 256)
        # Use the class-level cached model from FloodModelService to avoid
        # reloading the 400MB+ Keras model on every request.
        from app.services.flood_service import FloodModelService
        self.model = FloodModelService.vgg_model
        if self.model is None:
            raise RuntimeError(
                "VGG16 model is not loaded. Ensure FloodModelService.load_models() "
                "was called at startup."
            )

    def preprocess_image(self, image_path):
        """Load and preprocess a single image for prediction."""
        try:
            image = Image.open(image_path).convert("RGB")
            image = image.resize(self.img_size)
            image_array = np.array(image) / 255.0
            image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
            return image_array
        except Exception as e:
            raise RuntimeError(f"Error processing image '{image_path}': {e}")

    def predict(self, image_path):
        """
        Predict drain blockage class from image.

        Returns dict with keys:
          - blockage (int): CNN predicted class
              0 = Full blockage, 1 = No blockage, 2 = Partial blockage
          - probability (float): Confidence score for the predicted class
          - shap_values: Always None (CNN SHAP disabled — GradientExplainer too slow for VGG16)
        """
        image_array = self.preprocess_image(image_path)

        try:
            prediction = self.model.predict(image_array, verbose=0)
            predicted_class = int(np.argmax(prediction, axis=1)[0])
            probability = float(prediction[0, predicted_class])

            return {
                "blockage": predicted_class,
                "probability": probability,
                "shap_values": None,
            }
        except Exception as e:
            raise RuntimeError(f"Error during CNN prediction: {e}")
