# app/services/s3_service.py
import logging
import random
import base64
import boto3
from app.core.config import settings

logger = logging.getLogger(__name__)


class S3Service:
    def __init__(self):
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=settings.AWS_ACCESS_KEY,
            aws_secret_access_key=settings.AWS_SECRET_KEY,
        )
        self.bucket_name = settings.BUCKET_NAME
        self.folders = ["PARTIALLY BLOCKAGE/", "NO BLOCKAGE/", "FULL BLOCKAGE/"]

    def get_random_image_base64(self) -> dict:
        """
        Fetch a random image from a random S3 folder and return it as base64 string,
        along with its metadata (lat/lon guaranteed).
        """
        try:
            # Step 1: Pick a random folder
            random_folder = random.choice(self.folders)

            # Step 2: List objects in that folder
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=random_folder,
            )

            if "Contents" not in response or not response["Contents"]:
                raise RuntimeError(f"No images found in S3 folder '{random_folder}'")

            # Step 3: Filter out the folder key itself (0-byte prefix entry ends with "/")
            image_keys = [
                obj["Key"]
                for obj in response["Contents"]
                if not obj["Key"].endswith("/") and obj.get("Size", 0) > 0
            ]
            if not image_keys:
                raise RuntimeError(
                    f"No valid image files found in S3 folder '{random_folder}'"
                )

            image_key = random.choice(image_keys)

            # Step 4: Fetch the image
            image_obj = self.s3_client.get_object(
                Bucket=self.bucket_name, Key=image_key
            )
            image_data = image_obj["Body"].read()

            # Step 5: Fetch metadata for that image
            head_obj = self.s3_client.head_object(
                Bucket=self.bucket_name, Key=image_key
            )
            metadata = head_obj.get("Metadata", {})

            # Ensure lat/lon keys always exist
            lat = metadata.get("lat", "None")
            lon = metadata.get("lon", "None")

            return {
                "imageBase64": base64.b64encode(image_data).decode("utf-8"),
                "folder": random_folder,
                "imageKey": image_key,
                "metadata": {**metadata, "lat": lat, "lon": lon},
            }

        except Exception as e:
            raise RuntimeError(f"Failed to fetch image from S3: {e}")
