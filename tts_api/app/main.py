import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv

from app.api.routes import router as api_router

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Tacotron2 + HiFi-GAN Text-to-Speech API",
    description="API for high-quality speech synthesis using Tacotron2 and HiFi-GAN",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api")

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to the Tacotron2 + HiFi-GAN TTS API",
        "docs": "/docs",
        "endpoints": {
            "synthesize": "/api/tts/synthesize",
            "models": "/api/tts/models"
        }
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)