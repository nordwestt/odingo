from diffusers import DiffusionPipeline
from accelerate import cpu_offload
import torch
from typing import Optional, Dict, List
import os
import json
from pathlib import Path
import datetime

class ModelManager:
    def __init__(self, enable_offload: bool = True):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            print(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            print("CUDA not available, using CPU")
            
        self.models_dir = Path.home() / ".odingo" / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.models_info_path = self.models_dir / "models.json"
        self.loaded_models = {}
        self.default_model = "stable-diffusion-v1-5/stable-diffusion-v1-5"
        self.enable_offload = enable_offload
        self._load_models_info()
    
    def _load_models_info(self):
        if self.models_info_path.exists():
            with open(self.models_info_path, 'r') as f:
                self.models_info = json.load(f)
        else:
            self.models_info = {}
            self._save_models_info()
    
    def _save_models_info(self):
        with open(self.models_info_path, 'w') as f:
            json.dump(self.models_info, f, indent=2)
    
    async def install_model(self, model_id: str, name: Optional[str] = None) -> dict:
        """Install a model from Hugging Face Hub"""
        try:
            # Download the model
            pipeline = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            # Save model info
            model_info = {
                "name": name or model_id.split("/")[-1],
                "source": model_id,
                "installed_at": str(datetime.datetime.now()),
                "status": "ready"
            }
            
            self.models_info[model_id] = model_info
            self._save_models_info()
            
            return model_info
            
        except Exception as e:
            raise Exception(f"Failed to install model: {str(e)}")
    
    async def remove_model(self, model_id: str) -> bool:
        """Remove an installed model"""
        if model_id in self.loaded_models:
            del self.loaded_models[model_id]
        
        if model_id in self.models_info:
            del self.models_info[model_id]
            self._save_models_info()
            return True
        return False
    
    async def list_models(self) -> List[Dict]:
        """List all installed models"""
        return [
            {"id": model_id, **info}
            for model_id, info in self.models_info.items()
        ]
    
    async def get_pipeline(self, model_id: Optional[str] = None) -> DiffusionPipeline:
        model_id = model_id or self.default_model
        
        if model_id not in self.loaded_models:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            pipeline = DiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            if torch.cuda.is_available():
                if self.enable_offload:
                    pipeline.enable_model_cpu_offload()
                else:
                    pipeline = pipeline.to(self.device)
            
            self.loaded_models[model_id] = pipeline
            
        return self.loaded_models[model_id]
    
    async def generate_image(
        self,
        prompt: str,
        n: int = 1,
        steps: int = 10,
        size: tuple = (512, 512),
        model_id: Optional[str] = None
    ):
        print(f"Generating image with model {model_id}")
        pipeline = await self.get_pipeline(model_id)
        
        images = pipeline(
            prompt,
            num_images_per_prompt=n,
            height=size[1],
            width=size[0],
            num_inference_steps=steps,
        ).images
        
        return images 
