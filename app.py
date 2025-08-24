from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from main import generate_image_from_sketch
import os

app = FastAPI()

class ImageRequest(BaseModel):
    image_url: str

@app.post("/generate-image")
async def generate_image(request: ImageRequest):
    try:
        cloud_url = generate_image_from_sketch(request.image_url)
        return JSONResponse({"cloudinary_url": cloud_url})
    except Exception as e:
        print("ERROR:", e)
        return JSONResponse({"error": str(e)}, status_code=500)

from cloud_upload import upload_image_local

@app.post("/generate-image-file")
async def generate_image_file(file: UploadFile = File(...)):
    os.makedirs("static", exist_ok=True)
    file_location = os.path.join("static", file.filename)
    with open(file_location, "wb") as f:
        content = await file.read()
        f.write(content)
    try:
        cloud_url = upload_image_local(file_location)
        return {"filename": file.filename, "cloudinary_url": cloud_url}
    except Exception as e:
        print("Cloud upload failed:", e)
        return JSONResponse({"error": str(e)}, status_code=500)