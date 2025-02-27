# odingo - local image generation server

This is a local server for image generation, in the same way that [Ollama](https://ollama.ai/) is a local server for text generation, compliant with the OpenAI image generation API.

It uses Hugging Face's diffusers library to pull models from Hugging Face's model hub and generate images.

## To install

(maybe create venv by doing "python -m venv venv" and then doing "source venv/bin/activate")
pip install -r requirements.txt

## To run

uvicorn src.main:app --reload

## To use

### List installed models
curl http://localhost:8000/models/list

### Pull a model
curl -X POST http://localhost:8000/models/pull -d '{"url": "https://huggingface.co/CompVis/stable-diffusion-v1-4"}'

### Delete a model
curl -X DELETE http://localhost:8000/models/CompVis/stable-diffusion-v1-4

### Generate an image
curl -X POST http://localhost:8000/v1/images/generations -d '{"prompt": "A beautiful image of a cat", "n": 1, "size": "512x512", "model": "stable-diffusion-v1-5/stable-diffusion-v1-5", "response_format":"b64_json"}' --header "Content-Type: application/json"

