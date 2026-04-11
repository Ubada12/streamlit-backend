import numpy as np
from tensorflow import keras
from PIL import Image
from typing import Optional

from app.core.config import settings


class FloodPredictor:
    def __init__(self, model: Optional[object] = None):
        """
        Initialize the model.
        :param model_path: Path to the saved model (.keras)
        """
        self.img_size = (256, 256)
        self.model = model or self.load_model(settings.VGG16_MODEL_PATH)

    def load_model(self, model_path):
        """Load the trained CNN model from .keras file."""
        try:
            model = keras.models.load_model(model_path)
            print("Model loaded successfully.")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def preprocess_image(self, image_path):
        """Load and preprocess a single image for prediction."""
        try:
            image = Image.open(image_path).convert("RGB")
            image = image.resize(self.img_size)
            image_array = np.array(image) / 255.0
            image_array = np.expand_dims(image_array, axis=0)
            return image_array
        except Exception as e:
            print(f"Error processing image: {e}")
            return None

    def predict(self, image_path):
        """Predict the blockage class for a single image."""
        if self.model is None:
            return {"error": "Model is not loaded. Cannot make predictions."}

        image_array = self.preprocess_image(image_path)
        if image_array is None:
            return {"error": "Error processing image."}

        try:
            prediction = self.model.predict(image_array)
            predicted_class = int(np.argmax(prediction, axis=1)[0])
            probability = float(prediction[0, predicted_class])

            return {
                "blockage": predicted_class,
                "probability": probability,
                "shap_values": None,
            }
        except Exception as e:
            print(f"Error during prediction: {e}")
            return {"error": str(e)}
