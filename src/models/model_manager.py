from diffusers import StableDiffusionPipeline
from accelerate import cpu_offload
import torch
from typing import Optional
import os

class ModelManager:
    def __init__(self, enable_offload: bool = True):
        self.loaded_models = {}
        self.default_model = "stabilityai/stable-diffusion-2-1"
        self.enable_offload = enable_offload
        
    async def get_pipeline(self, model_id: Optional[str] = None) -> StableDiffusionPipeline:
        model_id = model_id or self.default_model
        
        if model_id not in self.loaded_models:
            pipeline = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            if torch.cuda.is_available():
                if self.enable_offload:
                    # Enable sequential CPU offloading
                    pipeline.enable_sequential_cpu_offload()
                    # Alternatively, for more fine-grained control:
                    # pipeline.enable_model_cpu_offload()
                else:
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