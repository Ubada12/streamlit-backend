# app/api/v1/endpoints/email.py
import logging
from fastapi import APIRouter, HTTPException, status
from app.models.email import FloodAlertEmailPayload
from app.services.email_service import EmailService

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/send-email", status_code=status.HTTP_200_OK)
async def send_email(payload: FloodAlertEmailPayload):
    """
    Send a XAI-FLOWS flood alert email via MSG91.

    Accepts a strictly validated payload containing all weather fields
    (with units pre-attached) and the detected flood risk level.
    The backend constructs the full MSG91 API payload internally.

    Only intended to be called when risk is High or Moderate —
    this is enforced by the frontend, but the endpoint itself
    does not reject Low/Minimal risk payloads to remain flexible.
    """
    try:
        result = await EmailService.send_flood_alert(payload)
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to send flood alert email: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
