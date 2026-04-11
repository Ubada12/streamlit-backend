# app/services/email_service.py
import httpx
from app.core.config import settings

class EmailService:
    @staticmethod
    async def send_email(payload: dict) -> dict:
        """
        Send email using MSG91 API.
        """
        if not payload or "recipients" not in payload:
            raise ValueError("Invalid email payload")

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.post(
                    settings.MSG91_API_URL,
                    headers={
                        "authkey": settings.MSG91_AUTH_KEY,
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"MSG91 API error: {e.response.text}")
        except Exception as e:
            raise RuntimeError(f"Failed to send email: {str(e)}")
