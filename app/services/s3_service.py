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
        self.folders = ["PARTIALLY BLOCKAGE/", "NO BLOCKAGE/", "FULL BLOCKAGE/"]

    def get_random_image_base64(self) -> str:
        """
        Fetch a random image from a random S3 folder and return it as base64 string.
        """
        try:
            # Step 1: Pick a random folder
            random_folder = random.choice(self.folders)

            # Step 2: List objects in that folder
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=random_folder
            )

            if 'Contents' not in response or not response['Contents']:
                raise RuntimeError(f"No images found in S3 folder '{random_folder}'")

            # Step 3: Pick a random image from that folder
            image_key = random.choice(response['Contents'])["Key"]

            # Step 4: Fetch and encode the image
            image_obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=image_key)
            image_data = image_obj['Body'].read()
            return {
                "imageBase64": base64.b64encode(image_data).decode('utf-8'),
                "folder": random_folder,
                "imageKey": image_key
                }
    
        except Exception as e:
            raise RuntimeError(f"Failed to fetch image from S3: {str(e)}")