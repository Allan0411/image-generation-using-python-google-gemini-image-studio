import cloudinary
import cloudinary.uploader
from cloudinary.utils import cloudinary_url
import os
from dotenv import load_dotenv

load_dotenv()

cloudinary.config( 
    cloud_name = os.getenv("CLOUDINARY_CLOUD_NAME"), 
    api_key = os.getenv("CLOUDINARY_API_KEY"), 
    api_secret = os.getenv("CLOUDINARY_API_SECRET"),
    secure=True
)

def upload_image_local(file_path, public_id=None):
    """Upload local file to Cloudinary."""
    result = cloudinary.uploader.upload(file_path, public_id=public_id)
    return result["secure_url"]

def upload_image_url(image_url, public_id=None):
    """Upload image from URL to Cloudinary."""
    result = cloudinary.uploader.upload(image_url, public_id=public_id)
    return result["secure_url"]

def optimize_url(public_id, width=None, height=None, crop="auto", quality="auto", fetch_format="auto"):
    url, _ = cloudinary_url(
        public_id,
        width=width,
        height=height,
        crop=crop,
        quality=quality,
        fetch_format=fetch_format
    )
    return url
