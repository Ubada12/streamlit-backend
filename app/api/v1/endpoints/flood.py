from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from app.services.flood_service import FloodModelService
from app.models.flood import FloodPredictionResponse
import logging
import traceback

router = APIRouter()

@router.post("/predict-flood/", response_model=FloodPredictionResponse)
async def predict_flood(image: UploadFile = File(...), request: str = Form(...)):
    """
    Predict flood risk using heuristic + ML pipeline.
    Returns detailed error messages if model loading or prediction fails.
    """
    try:
        # Ensure models are preloaded
        if not (FloodModelService.vgg_model and FloodModelService.xgb_model and FloodModelService.scaler):
            raise RuntimeError("Prediction models are not loaded. Please check model files or server startup.")

        # Perform prediction
        return FloodModelService.predict_flood(image, request)

    except FileNotFoundError as e:
        # Specific error if model file missing
        logging.error(f"Model file missing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Model file not found",
                "hint": "Ensure all required models are uploaded to /app/ml_models/"
            }
        )

    except ValueError as e:
        # Bad request data
        logging.warning(f"Invalid input: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": str(e)}
        )

    except Exception as e:
        # Log full traceback for debugging
        logging.error("Unexpected error in predict-flood:\n" + traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Internal server error during prediction",
                "details": str(e)
            }
        )
