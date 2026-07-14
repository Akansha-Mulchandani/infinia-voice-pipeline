"""
Common TTS interface for all voice cloning pipelines.

All pipeline implementations must inherit from TTSPipeline and implement
the required methods to ensure consistent evaluation across models.
"""

from abc import ABC, abstractmethod
from typing import Dict
import os


class TTSPipeline(ABC):
    """
    Abstract base class for TTS voice cloning pipelines.
    
    All language-specific pipelines must implement this interface to enable
    consistent evaluation and benchmarking.
    """
    
    def __init__(self, model_name: str, language: str):
        """
        Initialize the TTS pipeline.
        
        Args:
            model_name: Name of the model (e.g., "chatterbox", "xtts-v2")
            language: Language code (e.g., "en", "ar", "hi")
        """
        self.model_name = model_name
        self.language = language
        self.is_loaded = False
    
    @abstractmethod
    def load(self, reference_audio_path: str = None) -> None:
        """
        Load the model and any required resources.
        
        Args:
            reference_audio_path: Path to reference audio for voice cloning
                                 (some models need this at load time)
        """
        pass
    
    @abstractmethod
    def synthesize(self, text: str, reference_audio_path: str, out_path: str) -> Dict:
        """
        Synthesize speech from text using voice cloning.
        
        Args:
            text: Input text to synthesize
            reference_audio_path: Path to reference audio for voice cloning
            out_path: Path where output WAV file will be saved
        
        Returns:
            Dict containing:
                - audio_path: str, path to generated audio
                - gen_time_sec: float, generation time in seconds
                - sample_rate: int, sample rate of generated audio
                - success: bool, whether synthesis succeeded
                - error: str, error message if success=False
        """
        pass
    
    def validate_inputs(self, text: str, reference_audio_path: str, out_path: str) -> None:
        """
        Validate input parameters before synthesis.
        
        Raises:
            ValueError: If inputs are invalid
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        if not os.path.exists(reference_audio_path):
            raise ValueError(f"Reference audio not found: {reference_audio_path}")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dict with model metadata
        """
        return {
            "model_name": self.model_name,
            "language": self.language,
            "is_loaded": self.is_loaded
        }
