from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from ..models.model_manager import ModelManager
import base64
import time
from io import BytesIO
from pathlib import Path
import uuid
import os
from datetime import datetime, timedelta

router = APIRouter()
model_manager = ModelManager()

IMAGES_DIR = Path("static/images")
IMAGES_URL_BASE = "http://localhost:8000/static/images"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_EXPIRY_MINUTES = 10


class ImageGenerationRequest(BaseModel):
    prompt: str
    n: Optional[int] = 1
    size: Optional[str] = "1024x1024"
    response_format: Optional[str] = "url"
    model: Optional[str] = None
    steps: Optional[int] = 10

class ImageResponse(BaseModel):
    created: int
    data: List[dict]

class ModelInstallRequest(BaseModel):
    name: Optional[str] = None
    url: str

class ModelResponse(BaseModel):
    id: str
    name: str
    source: str
    installed_at: str
    status: str

def cleanup_expired_images():
    current_time = datetime.now()
    for file in IMAGES_DIR.glob("*.png"):
        # Get creation time and check if expired
        creation_time = datetime.fromtimestamp(file.stat().st_ctime)
        if current_time - creation_time > timedelta(minutes=IMAGE_EXPIRY_MINUTES):
            try:
                file.unlink()  # Delete the file
            except Exception as e:
                print(f"Failed to delete {file}: {e}")

@router.post("/v1/images/generations", response_model=ImageResponse)
async def create_image(request: ImageGenerationRequest):
    try:
        cleanup_expired_images()
        # Parse size
        width, height = map(int, request.size.split("x"))

        # if no steps, set to 10
        if request.steps is None:
            request.steps = 10
        
        # Generate images
        images = await model_manager.generate_image(
            prompt=request.prompt,
            n=request.n,
            size=(width, height),
            model_id=request.model,
            steps=request.steps
        )
        
        # Convert images to response format
        response_data = []
        for img in images:
            if request.response_format == "b64_json":
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                response_data.append({"b64_json": img_str})
            else:
                # Generate a unique filename
                filename = f"{uuid.uuid4()}.png"
                filepath = IMAGES_DIR / filename
                
                # Save the image
                img.save(filepath)
                
                # Create the public URL
                image_url = f"{IMAGES_URL_BASE}/{filename}"
                response_data.append({"url": image_url})
        
        return ImageResponse(
            created=int(time.time()),
            data=response_data
        )
    
    except Exception as e:
        if(e.args[0].startswith("CUDA out of memory")):
            raise HTTPException(status_code=500, detail="CUDA out of memory. Try reducing the number of images or steps.")
        raise HTTPException(status_code=500, detail=str(e)) 

@router.post("/models/pull", response_model=ModelResponse)
async def pull_model(request: ModelInstallRequest):
    """Install a new model from Hugging Face Hub"""
    try:
        model_info = await model_manager.install_model(request.url, request.name)
        return ModelResponse(id=request.url, **model_info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/models/{model_id}")
async def delete_model(model_id: str):
    """Remove an installed model"""
    success = await model_manager.remove_model(model_id)
    if not success:
        raise HTTPException(status_code=404, detail="Model not found")
    return {"status": "success", "message": f"Model {model_id} removed"}

@router.get("/models/list", response_model=List[ModelResponse])
async def list_models():
    """List all installed models"""
    models = await model_manager.list_models()
    return models 
