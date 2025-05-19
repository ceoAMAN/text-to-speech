from fastapi import APIRouter, HTTPException, Query, BackgroundTasks, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import io
import torch
import numpy as np
import uuid
import os
import time
from typing import List, Optional

from app.models.tacotron2 import Tacotron2TTS
from app.models.hifigan import HiFiGANVocoder
from app.utils.audio_helpers import save_wav, audio_bytes_from_array

# Initialize router
router = APIRouter(prefix="/tts", tags=["Text-to-Speech"])

# Create model instances
tacotron2_model = Tacotron2TTS()
hifigan_model = HiFiGANVocoder()

# Cache directory for audio files
CACHE_DIR = os.environ.get("CACHE_DIR", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# Define request models
class TTSRequest(BaseModel):
    text: str = Field(..., example="The quick brown fox jumps over the lazy dog.", 
                      description="Text to synthesize into speech")
    speed: float = Field(1.0, ge=0.5, le=2.0, 
                         description="Speech rate adjustment factor (0.5-2.0)")
    pitch: float = Field(0.0, ge=-10.0, le=10.0, 
                         description="Pitch adjustment in semitones (-10 to 10)")
    energy: float = Field(1.0, ge=0.5, le=1.5, 
                          description="Energy/volume adjustment factor (0.5-1.5)")

# Available audio formats
AUDIO_FORMATS = ["wav", "mp3", "ogg"]

@router.post("/synthesize")
async def synthesize_speech(
    request: TTSRequest,
    format: str = Query("wav", enum=AUDIO_FORMATS),
    background_tasks: BackgroundTasks
):
    """
    Synthesize speech from text using Tacotron2 and HiFi-GAN
    """
    try:
        start_time = time.time()
        
        # Generate mel spectrograms with Tacotron2
        mel_outputs, mel_lengths, alignment = tacotron2_model.infer(
            request.text, 
            speed=request.speed,
            pitch=request.pitch
        )
        
        # Convert mel spectrograms to audio with HiFi-GAN
        audio = hifigan_model.infer(
            mel_outputs, 
            energy_factor=request.energy
        )
        
        inference_time = time.time() - start_time
        
        # Create a unique filename for the audio
        filename = f"{uuid.uuid4()}.{format}"
        filepath = os.path.join(CACHE_DIR, filename)
        
        # Get audio as bytes in the specified format
        audio_bytes = audio_bytes_from_array(audio, format=format)
        
        # Store file in cache for potential later retrieval
        background_tasks.add_task(
            save_wav, 
            audio, 
            filepath, 
            sample_rate=hifigan_model.sampling_rate
        )
        
        # Set content type based on format
        content_type = {
            "wav": "audio/wav",
            "mp3": "audio/mpeg",
            "ogg": "audio/ogg",
        }.get(format, "audio/wav")
        
        headers = {
            "X-Processing-Time": str(inference_time),
            "X-Audio-Duration": str(len(audio) / hifigan_model.sampling_rate),
            "X-Cache-ID": filename
        }
        
        # Return the audio as a streaming response
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type=content_type,
            headers=headers
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {str(e)}")

@router.get("/models")
async def get_models_info():
    """Get information about the loaded TTS models"""
    try:
        return {
            "tacotron2": {
                "name": tacotron2_model.model_name,
                "version": tacotron2_model.version,
                "params": tacotron2_model.params_count,
                "loaded": tacotron2_model.is_loaded,
            },
            "hifigan": {
                "name": hifigan_model.model_name,
                "version": hifigan_model.version,
                "params": hifigan_model.params_count,
                "loaded": hifigan_model.is_loaded,
                "sampling_rate": hifigan_model.sampling_rate
            },
            "default_settings": {
                "speed": 1.0,
                "pitch": 0.0,
                "energy": 1.0,
                "format": "wav"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

@router.get("/cached/{filename}")
async def get_cached_audio(filename: str):
    """Retrieve a previously generated audio file from cache"""
    filepath = os.path.join(CACHE_DIR, filename)
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    format = filename.split(".")[-1]
    content_type = {
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "ogg": "audio/ogg",
    }.get(format, "application/octet-stream")
    
    def iterfile():
        with open(filepath, "rb") as f:
            yield from f
    
    return StreamingResponse(
        iterfile(),
        media_type=content_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )