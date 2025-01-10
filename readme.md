# ODingo - local image generation server

This is a local server for image generation, in the same way that [Ollama](https://ollama.ai/) is a local server for text generation, compliant with the OpenAI image generation API.

It uses Hugging Face's diffusers library to pull models from Hugging Face's model hub and generate images.

## To install

pip install -r requirements.txt

## To run

uvicorn src.main:app --reload