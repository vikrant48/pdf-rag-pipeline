import os
import boto3
from botocore.client import Config
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

ENDPOINT_URL = os.getenv("SUPABASE_S3_ENDPOINT")
ACCESS_KEY = os.getenv("SUPABASE_S3_ACCESS_KEY")
SECRET_KEY = os.getenv("SUPABASE_S3_SECRET_KEY")
BUCKET = os.getenv("SUPABASE_S3_BUCKET", "RAG_DOCUMENTS")
PUBLIC_URL_PREFIX = os.getenv("SUPABASE_S3_PUBLIC_URL_PREFIX")

def get_s3_client():
    """Returns a boto3 S3 client configured for Supabase storage."""
    if not (ENDPOINT_URL and ACCESS_KEY and SECRET_KEY):
        logger.warning("Supabase S3 credentials are not fully defined in .env. Falling back to local storage.")
        return None
    try:
        s3 = boto3.client(
            's3',
            endpoint_url=ENDPOINT_URL,
            aws_access_key_id=ACCESS_KEY,
            aws_secret_access_key=SECRET_KEY,
            config=Config(signature_version='s3v4'),
            region_name='us-east-1' # Default region string required by boto3 for custom endpoints
        )
        return s3
    except Exception as e:
        logger.error(f"Error creating boto3 S3 client: {e}")
        return None

def upload_pdf_to_s3(file_bytes: bytes, filename: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Uploads document bytes to the Supabase S3 bucket.
    Returns a tuple of (s3_key, public_url).
    """
    s3_client = get_s3_client()
    if not s3_client:
        logger.warning(f"S3 client not initialized. PDF {filename} will only be stored locally.")
        return None, None
        
    s3_key = filename
    try:
        logger.info(f"Uploading '{filename}' to S3 bucket '{BUCKET}'...")
        s3_client.put_object(
            Bucket=BUCKET,
            Key=s3_key,
            Body=file_bytes,
            ContentType='application/pdf'
        )
        public_url = f"{PUBLIC_URL_PREFIX}/{s3_key}"
        logger.info(f"Upload complete. S3 Key: '{s3_key}' | Public URL: '{public_url}'")
        return s3_key, public_url
    except Exception as e:
        logger.error(f"Failed to upload '{filename}' to S3: {e}")
        return None, None
