import numpy as np
import scipy.io.wavfile
import io
import librosa
import soundfile as sf
from typing import Union, Optional
import logging

logger = logging.getLogger(__name__)

def save_wav(
    audio: np.ndarray, 
    path_or_buf: Union[str, io.BytesIO], 
    sample_rate: int = 22050,
    return_buffer: bool = False
) -> Union[float, io.BytesIO]:
    """
    Save audio to WAV file or buffer.
    
    Args:
        audio: Audio data as numpy array
        path_or_buf: File path or BytesIO buffer
        sample_rate: Audio sample rate
        return_buffer: Whether to return the buffer object
        
    Returns:
        Audio length in seconds if path is given, or buffer if return_buffer=True
    """
    try:
        # Normalize audio if needed (to prevent clipping)
        if np.abs(audio).max() > 1.0:
            audio = audio / np.abs(audio).max()
        
        # Convert to int16 format
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Save to file or buffer
        if isinstance(path_or_buf, str):
            scipy.io.wavfile.write(path_or_buf, sample_rate, audio_int16)
            audio_length = len(audio) / sample_rate
            logger.info(f"Saved audio to {path_or_buf} ({audio_length:.2f}s)")
            return audio_length
        else:
            # Writing to BytesIO buffer
            scipy.io.wavfile.write(path_or_buf, sample_rate, audio_int16)
            if return_buffer:
                return path_or_buf
            else:
                return len(audio) / sample_rate
    
    except Exception as e:
        logger.error(f"Error saving audio: {e}", exc_info=True)
        raise

def convert_to_mp3(
    wav_path: str, 
    mp3_path: str, 
    bitrate: str = "192k"
) -> str:
    """
    Convert WAV file to MP3.
    
    Args:
        wav_path: Path to input WAV file
        mp3_path: Path to output MP3 file
        bitrate: MP3 bitrate
        
    Returns:
        Path to the output MP3 file
    """
    try:
        import ffmpeg
        
        # Convert WAV to MP3 using FFmpeg
        (
            ffmpeg
            .input(wav_path)
            .output(mp3_path, audio_bitrate=bitrate)
            .run(quiet=True, overwrite_output=True)
        )
        
        logger.info(f"Converted {wav_path} to MP3: {mp3_path}")
        return mp3_path
    
    except ImportError:
        logger.warning("FFmpeg not available, using librosa for MP3 conversion (lower quality)")
        
        # Alternative method using librosa/soundfile
        y, sr = librosa.load(wav_path, sr=None)
        sf.write(mp3_path, y, sr, format='mp3')
        
        logger.info(f"Converted {wav_path} to MP3: {mp3_path} (using librosa)")
        return mp3_path
    
    except Exception as e:
        logger.error(f"Error converting to MP3: {e}", exc_info=True)
        raise

def extract_audio_features(
    audio: np.ndarray, 
    sample_rate: int = 22050
) -> dict:
    """
    Extract audio features for analysis.
    
    Args:
        audio: Audio data as numpy array
        sample_rate: Audio sample rate
        
    Returns:
        Dictionary of audio features
    """
    try:
        # Extract features
        duration = len(audio) / sample_rate
        rms = np.sqrt(np.mean(audio**2))
        
        # Zero-crossing rate
        zero_crossings = np.sum(np.abs(np.diff(np.signbit(audio)))) / len(audio)
        
        # Spectral centroid
        cent = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0].mean()
        
        return {
            "duration": duration,
            "rms_amplitude": float(rms),
            "zero_crossing_rate": float(zero_crossings),
            "spectral_centroid": float(cent)
        }
    
    except Exception as e:
        logger.error(f"Error extracting audio features: {e}", exc_info=True)
        raise

def adjust_audio_speed(
    audio: np.ndarray, 
    speed_factor: float = 1.0
) -> np.ndarray:
    """
    Adjust audio playback speed.
    
    Args:
        audio: Audio data as numpy array
        speed_factor: Speed factor (0.5 = half speed, 2.0 = double speed)
        
    Returns:
        Speed-adjusted audio
    """
    if speed_factor == 1.0:
        return audio
    
    try:
        # Use librosa's time stretch
        return librosa.effects.time_stretch(audio, rate=speed_factor)
    
    except Exception as e:
        logger.error(f"Error adjusting audio speed: {e}", exc_info=True)
        raise

def apply_fade(
    audio: np.ndarray, 
    fade_in_sec: float = 0.01, 
    fade_out_sec: float = 0.01,
    sample_rate: int = 22050
) -> np.ndarray:
    """
    Apply fade-in and fade-out to audio.
    
    Args:
        audio: Audio data as numpy array
        fade_in_sec: Fade-in duration in seconds
        fade_out_sec: Fade-out duration in seconds
        sample_rate: Audio sample rate
        
    Returns:
        Audio with fades applied
    """
    try:
        # Apply fade-in
        fade_in_samples = int(fade_in_sec * sample_rate)
        if fade_in_samples > 0:
            fade_in = np.linspace(0, 1, fade_in_samples)
            audio[:fade_in_samples] = audio[:fade_in_samples] * fade_in
        
        # Apply fade-out
        fade_out_samples = int(fade_out_sec * sample_rate)
        if fade_out_samples > 0:
            fade_out = np.linspace(1, 0, fade_out_samples)
            audio[-fade_out_samples:] = audio[-fade_out_samples:] * fade_out
        
        return audio
    
    except Exception as e:
        logger.error(f"Error applying audio fades: {e}", exc_info=True)
        raise