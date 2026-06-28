# app/services/email_service.py
import logging
import httpx
from app.core.config import settings
from app.models.email import FloodAlertEmailPayload

logger = logging.getLogger(__name__)


class EmailService:

    @staticmethod
    def _build_msg91_payload(payload: FloodAlertEmailPayload) -> dict:
        """
        Construct the full MSG91 email API payload from a validated
        FloodAlertEmailPayload. MSG91 uses template variables that map
        directly to the ##placeholder## tokens in the HTML template.
        """
        return {
            "template_id": settings.MSG91_TEMPLATE_ID,
            "from": {
                "name": settings.SENDER_NAME,
                "email": settings.SENDER_EMAIL,
            },
            "recipients": [
                {
                    "to": [
                        {
                            "email": settings.RECIPIENT_EMAIL,
                            "name": settings.RECIPIENT_NAME,
                        }
                    ],
                    # Keys must match ##placeholder## names in the MSG91 template
                    "variables": {
                        "Timestamp": payload.Timestamp,
                        "risk": payload.risk,
                        "address": payload.address,
                        "latitude": payload.latitude,
                        "longitude": payload.longitude,
                        "temperature": payload.temperature,
                        "precipitation": payload.precipitation,
                        "wind_speed": payload.wind_speed,
                        "humidity": payload.humidity,
                        "visibility": payload.visibility,
                        "condition": payload.condition,
                        "uv_index": payload.uv_index,
                        "pressure": payload.pressure,
                        "cloud_coverage": payload.cloud_coverage,
                    },
                }
            ],
        }

    @staticmethod
    async def send_flood_alert(payload: FloodAlertEmailPayload) -> dict:
        """
        Send a flood alert email via MSG91 using the provided payload.
        Constructs the MSG91 API structure internally — callers only
        need to supply the XAI-FLOWS domain fields.
        """
        msg91_payload = EmailService._build_msg91_payload(payload)

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.post(
                    settings.MSG91_API_URL,
                    headers={
                        "authkey": settings.MSG91_AUTH_KEY,
                        "Content-Type": "application/json",
                    },
                    json=msg91_payload,
                )
                response.raise_for_status()
                logger.info(
                    f"Flood alert email sent — risk={payload.risk}, "
                    f"address={payload.address}"
                )
                return response.json()

        except httpx.HTTPStatusError as e:
            raise RuntimeError(
                f"MSG91 API error {e.response.status_code}: {e.response.text}"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to send flood alert email: {e}")
