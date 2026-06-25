import logging
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class FloodPredictor:
    def __init__(self, model):
        """
        Initialize with a pre-loaded Keras model instance.
        Model is passed in by the caller (FloodModelService) to avoid
        circular imports and repeated disk reads.
        """
        self.img_size = (256, 256)
        self.model    = model
        if self.model is None:
            raise RuntimeError(
                "VGG16 model is None — ensure FloodModelService.load_models() ran at startup."
            )

    def preprocess_image(self, image_path):
        """Load and preprocess a single image for prediction."""
        try:
            image       = Image.open(image_path).convert("RGB")
            image       = image.resize(self.img_size)
            image_array = np.array(image) / 255.0
            image_array = np.expand_dims(image_array, axis=0)
            return image_array
        except Exception as e:
            raise RuntimeError(f"Error processing image '{image_path}': {e}")

    def predict(self, image_path):
        """
        Predict drain blockage class from image.

        Returns dict with keys:
          - blockage (int): 0=Full blockage, 1=No blockage, 2=Partial blockage
          - probability (float): Confidence score for the predicted class
          - shap_values: Always None (CNN SHAP disabled — too slow for VGG16 at inference)
        """
        image_array = self.preprocess_image(image_path)
        try:
            prediction     = self.model.predict(image_array, verbose=0)
            predicted_class = int(np.argmax(prediction, axis=1)[0])
            probability     = float(prediction[0, predicted_class])
            return {
                "blockage":    predicted_class,
                "probability": probability,
                "shap_values": None,
            }
        except Exception as e:
            raise RuntimeError(f"Error during CNN prediction: {e}")
