# app/api/v1/router.py
from fastapi import APIRouter
from app.api.v1.endpoints import flood, weather, s3, email

api_router = APIRouter()

api_router.include_router(flood.router, tags=["Flood"])
api_router.include_router(weather.router, tags=["Weather"])
api_router.include_router(s3.router, tags=["S3"])
api_router.include_router(email.router, tags=["Email"])
