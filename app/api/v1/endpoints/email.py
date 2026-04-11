# app/api/v1/endpoints/email.py
from fastapi import APIRouter, Request, HTTPException
from app.services.email_service import EmailService

router = APIRouter()

@router.post("/send-email")
async def send_email(request: Request):
    """
    Send email via MSG91 API with JSON payload.
    """
    try:
        body = await request.json()
        result = await EmailService.send_email(body)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
