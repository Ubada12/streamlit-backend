# app/api/v1/endpoints/s3.py
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from app.services.s3_service import S3Service

router = APIRouter()

@router.get("/get-latest-s3-image")
async def get_latest_s3_image():
    """
    Retrieve a random image from S3 and return it as base64.
    """
    try:
        service = S3Service()
        encoded_image = service.get_random_image_base64()
        return JSONResponse(content={"imageBase64": encoded_image})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
