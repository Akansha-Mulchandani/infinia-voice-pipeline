"""
Audio utilities for TTS pipelines.

Common functions for loading, validating, and processing audio files.
"""

import os
import soundfile as sf
import numpy as np
import librosa


def load_audio(audio_path: str, target_sr: int = 16000) -> tuple:
    """
    Load audio file and resample to target sample rate.
    
    Args:
        audio_path: Path to audio file
        target_sr: Target sample rate (default: 16000)
    
    Returns:
        Tuple of (audio_array, sample_rate)
    
    Raises:
        ValueError: If audio file is invalid or empty
    """
    if not os.path.exists(audio_path):
        raise ValueError(f"Audio file not found: {audio_path}")
    
    try:
        audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
    except Exception as e:
        raise ValueError(f"Failed to load audio file: {e}")
    
    if len(audio) == 0:
        raise ValueError(f"Audio file is empty: {audio_path}")
    
    return audio, sr


def validate_audio(audio_path: str, min_duration: float = 1.0, max_duration: float = 60.0) -> dict:
    """
    Validate audio file meets requirements.
    
    Args:
        audio_path: Path to audio file
        min_duration: Minimum duration in seconds
        max_duration: Maximum duration in seconds
    
    Returns:
        Dict with validation results
    
    Raises:
        ValueError: If validation fails
    """
    try:
        audio, sr = load_audio(audio_path)
        duration = len(audio) / sr
        
        if duration < min_duration:
            raise ValueError(f"Audio too short: {duration:.2f}s < {min_duration}s")
        
        if duration > max_duration:
            raise ValueError(f"Audio too long: {duration:.2f}s > {max_duration}s")
        
        return {
            "valid": True,
            "duration": duration,
            "sample_rate": sr,
            "channels": 1,  # We always load as mono
            "samples": len(audio)
        }
    except Exception as e:
        return {
            "valid": False,
            "error": str(e)
        }


def save_audio(audio: np.ndarray, out_path: str, sample_rate: int = 16000) -> None:
    """
    Save audio array to WAV file.
    
    Args:
        audio: Audio array (numpy array)
        out_path: Output file path
        sample_rate: Sample rate (default: 16000)
    
    Raises:
        ValueError: If audio is invalid
    """
    if len(audio) == 0:
        raise ValueError("Audio array is empty")
    
    # Ensure output directory exists (skip if out_path is a bare
    # filename with no directory component, e.g. from eval scripts)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    
    # Save as WAV
    sf.write(out_path, audio, sample_rate)


def get_audio_duration(audio_path: str) -> float:
    """
    Get duration of audio file in seconds.
    
    Args:
        audio_path: Path to audio file
    
    Returns:
        Duration in seconds
    """
    try:
        audio, sr = load_audio(audio_path)
        return len(audio) / sr
    except Exception:
        return 0.0


def normalize_audio(audio: np.ndarray, target_level: float = -3.0) -> np.ndarray:
    """
    Normalize audio to target dB level.
    
    Args:
        audio: Audio array
        target_level: Target level in dB (default: -3.0)
    
    Returns:
        Normalized audio array
    """
    if len(audio) == 0:
        return audio
    
    # Calculate current RMS
    rms = np.sqrt(np.mean(audio ** 2))
    
    if rms == 0:
        return audio
    
    # Calculate scaling factor
    target_rms = 10 ** (target_level / 20)
    scale = target_rms / rms
    
    # Apply scaling
    normalized = audio * scale
    
    # Clip to prevent distortion
    normalized = np.clip(normalized, -1.0, 1.0)
    
    return normalized
