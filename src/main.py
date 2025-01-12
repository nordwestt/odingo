from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api.routes import router
from fastapi.staticfiles import StaticFiles


app = FastAPI(
    title="ODingo",
    description="Local image generation server compatible with OpenAI's API",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(router)

@app.get("/")
async def root():
    return {"message": "Welcome to ODingo - Local Image Generation Server"} 