# app/services/s3_service.py
import boto3, random, base64
from app.core.config import settings

class S3Service:
    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.AWS_ACCESS_KEY,
            aws_secret_access_key=settings.AWS_SECRET_KEY
        )
        self.bucket_name = settings.BUCKET_NAME
        self.folder_prefix = "NO BLOCKAGE/"

    def get_random_image_base64(self) -> str:
        """
        Fetch a random image from S3 folder and return it as base64 string.
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name, Prefix=self.folder_prefix
            )

            if 'Contents' not in response or not response['Contents']:
                raise RuntimeError("No images found in S3 bucket")

            # Randomly select an image
            image_key = random.choice(response['Contents'])["Key"]

            # Download and encode to base64
            image_obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=image_key)
            image_data = image_obj['Body'].read()
            return base64.b64encode(image_data).decode('utf-8')

        except Exception as e:
            raise RuntimeError(f"Failed to fetch image from S3: {str(e)}")
