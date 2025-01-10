from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from ..models.model_manager import ModelManager
import base64
from io import BytesIO

router = APIRouter()
model_manager = ModelManager()

class ImageGenerationRequest(BaseModel):
    prompt: str
    n: Optional[int] = 1
    size: Optional[str] = "1024x1024"
    response_format: Optional[str] = "url"
    model: Optional[str] = None

class ImageResponse(BaseModel):
    created: int
    data: List[dict]

@router.post("/v1/images/generations", response_model=ImageResponse)
async def create_image(request: ImageGenerationRequest):
    try:
        # Parse size
        width, height = map(int, request.size.split("x"))
        
        # Generate images
        images = await model_manager.generate_image(
            prompt=request.prompt,
            n=request.n,
            size=(width, height),
            model_id=request.model
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
                # In a real implementation, you'd save the image and create a URL
                # For now, we'll just return a placeholder
                response_data.append({"url": "http://localhost:8000/images/generated.png"})
        
        return ImageResponse(
            created=int(time.time()),
            data=response_data
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 