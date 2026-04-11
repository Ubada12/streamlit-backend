import json
import os
import pickle
import shutil
import tempfile

from tensorflow.keras.models import load_model

from app.core.config import settings
from app.models.flood import FloodPredictionRequest, FloodPredictionResponse
from app.utils.heuristic_rule import HeuristicModel


class FloodModelService:
    vgg_model = None
    xgb_model = None
    scaler = None

    @classmethod
    def load_models(cls):
        """
        Load all models once into memory at startup.
        Safe to call multiple times.
        """
        if cls.vgg_model is None:
            try:
                cls.vgg_model = load_model(settings.VGG16_MODEL_PATH)
                print("VGG16 model loaded successfully.")
            except Exception as e:
                print(f"Error loading VGG16 model: {e}")

        if cls.xgb_model is None:
            try:
                with open(settings.XGB_MODEL_PATH, "rb") as f:
                    cls.xgb_model = pickle.load(f)
                print("XGBoost model loaded successfully.")
            except Exception as e:
                print(f"Error loading XGBoost model: {e}")

        if cls.scaler is None:
            try:
                with open(settings.SCALER_PATH, "rb") as f:
                    cls.scaler = pickle.load(f)
                print("Scaler loaded successfully.")
            except Exception as e:
                print(f"Error loading Scaler: {e}")

    @staticmethod
    def predict_flood(image_file, request_json: str) -> FloodPredictionResponse:
        """
        Run flood prediction pipeline using cached models.
        """
        temp_path = None
        try:
            request_data = json.loads(request_json)
            request_model = FloodPredictionRequest(**request_data)

            suffix = os.path.splitext(image_file.filename or "")[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_path = temp_file.name
                shutil.copyfileobj(image_file.file, temp_file)

            heuristic_model = HeuristicModel(
                image_path=temp_path,
                lon=request_model.lon,
                lat=request_model.lat,
                vgg_model=FloodModelService.vgg_model,
                xgb_model=FloodModelService.xgb_model,
                scaler=FloodModelService.scaler,
            )
            result = heuristic_model.predict()

            weather_shap_value = (
                [
                    {"feature": f, "value": v}
                    for f, v in zip(
                        heuristic_model.weather_shap_value.feature_names,
                        heuristic_model.weather_shap_value.values.tolist()[0],
                    )
                ]
                if hasattr(heuristic_model.weather_shap_value, "values")
                else None
            )
            drain_blockage_shap_value = (
                heuristic_model.blockage_shap_value.values.tolist()
                if hasattr(heuristic_model.blockage_shap_value, "values")
                else None
            )

            return FloodPredictionResponse(
                image=image_file.filename,
                longitude=request_model.lon,
                latitude=request_model.lat,
                prediction=result,
                weather_data=heuristic_model.input_data,
                weather_prediction=heuristic_model.output_data,
                weather_metadata=heuristic_model.metadata,
                weather_shap_value=weather_shap_value,
                drain_blockage=int(heuristic_model.blockage),
                drain_blockage_prob=float(heuristic_model.blockage_prob),
                drain_blockage_shape_value=drain_blockage_shap_value,
            )

        except Exception as e:
            raise RuntimeError(f"Flood prediction failed: {str(e)}") from e
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
