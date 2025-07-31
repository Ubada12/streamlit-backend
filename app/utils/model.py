import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import shap
from app.core.config import settings

class FloodPredictor:
    def __init__(self):
        """
        Initialize the model.
        :param model_path: Path to the saved model (.keras)
        """
        self.img_size = (256, 256)  # Define image size before model initialization
        self.model = self.load_model(settings.VGG16_MODEL_PATH)
        # self.explainer = None  # SHAP Explainer will be initialized later

        # if self.model:
        #     # Initialize SHAP only if the model was successfully loaded
        #     background = np.random.rand(1, *self.img_size, 3)  # Random background sample
        #     self.explainer = shap.GradientExplainer(self.model, background)
        #     print("✅ SHAP Explainer initialized.")

    def load_model(self, model_path):
        """Load the trained CNN model from .keras file."""
        try:
            model = keras.models.load_model(model_path)
            print("✅ Model loaded successfully.")
            return model
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None

    def preprocess_image(self, image_path):
        """Load and preprocess a single image for prediction."""
        try:
            image = Image.open(image_path).convert("RGB")
            image = image.resize(self.img_size)  # Resize to match CNN input size
            image_array = np.array(image) / 255.0  # Normalize pixel values
            image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
            return image_array
        except Exception as e:
            print(f"❌ Error processing image: {e}")
            return None

    def predict(self, image_path):
        """Predict whether the image contains a blockage and return SHAP values."""
        if self.model is None:
            return "⚠️ Model is not loaded. Cannot make predictions."

        image_array = self.preprocess_image(image_path)
        if image_array is None:
            return "⚠️ Error processing image."
        
        try:
            # Get model prediction
            prediction = self.model.predict(image_array)
            predicted_class = np.argmax(prediction, axis=1)[0]
            probability = prediction[0, predicted_class]

            # Compute SHAP values
            # shap_values = None
            # if self.explainer:
            #     shap_values = self.explainer.shap_values(image_array)

            return {
                "blockage": predicted_class,
                "probability": probability,
                # "shap_values": shap_values if shap_values else "SHAP computation failed"
            }
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            return {"error": str(e)}
