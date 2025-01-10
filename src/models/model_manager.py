from diffusers import StableDiffusionPipeline
import torch
from typing import Optional
import os

class ModelManager:
    def __init__(self):
        self.loaded_models = {}
        self.default_model = "stabilityai/stable-diffusion-2-1"
        
    async def get_pipeline(self, model_id: Optional[str] = None) -> StableDiffusionPipeline:
        model_id = model_id or self.default_model
        
        if model_id not in self.loaded_models:
            pipeline = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            if torch.cuda.is_available():
                pipeline = pipeline.to("cuda")
            
            self.loaded_models[model_id] = pipeline
            
        return self.loaded_models[model_id]
    
    async def generate_image(
        self,
        prompt: str,
        n: int = 1,
        size: tuple = (512, 512),
        model_id: Optional[str] = None
    ):
        pipeline = await self.get_pipeline(model_id)
        
        images = pipeline(
            prompt,
            num_images_per_prompt=n,
            height=size[1],
            width=size[0]
        ).images
        
        return images 