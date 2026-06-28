# app/api/v1/endpoints/flood.py
import logging
import traceback
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from app.services.flood_service import FloodModelService
from app.models.flood import FloodPredictionResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/predict-flood/", response_model=FloodPredictionResponse)
async def predict_flood(image: UploadFile = File(...), request: str = Form(...)):
    """
    Unified flood prediction endpoint.

    Accepts a drain-camera image and coordinates, runs the full ML pipeline,
    reverse-geocodes the location, dispatches an alert email on High/Moderate
    risk, and returns a single clean response.

    All error detail values are plain strings so the frontend renders them directly.
    """
    try:
        if not (
            FloodModelService.vgg_model
            and FloodModelService.xgb_model
            and FloodModelService.scaler
        ):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Prediction models are not loaded. Please check server startup logs.",
            )
        return await FloodModelService.predict_flood(image, request)

    except HTTPException:
        raise

    except FileNotFoundError as e:
        logger.error(f"Model file missing: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model file not found: {e}. Ensure all model files exist in /ml_models/.",
        )

    except ValueError as e:
        logger.warning(f"Invalid input to predict-flood: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )

    except Exception as e:
        logger.error("Unexpected error in predict-flood:\n" + traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
