"""
Arabic voice cloning pipeline using XTTS-v2 (Coqui TTS).

XTTS-v2 is a multilingual TTS model with voice cloning capabilities.
"""

import sys
import os
import time
import argparse

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))

from tts_interface import TTSPipeline
from audio_utils import load_audio, save_audio, validate_audio


class XTTSArabicPipeline(TTSPipeline):
    """
    XTTS-v2 voice cloning pipeline for Arabic.
    
    NOTE: Using the maintained fork of Coqui TTS since coqui-ai/TTS is archived.
    Need to verify Arabic language support in the installed version.
    """
    
    def __init__(self):
        super().__init__(model_name="xtts-v2", language="ar")
        self.model = None
        self.tts = None
    
    def load(self, reference_audio_path: str = None) -> None:
        """
        Load XTTS-v2 model.
        
        Args:
            reference_audio_path: Path to reference audio (may not be needed at load time)
        """
        try:
            # Import TTS from Coqui
            from TTS.api import TTS
            
            # Initialize XTTS-v2 model
            # NOTE: Check if Arabic is supported in the model
            # Common model name: "tts_models/multilingual/multi-dataset/xtts_v2"
            self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
            
            # Move to GPU if available
            self.tts.to("cuda" if self.tts.is_cuda_available else "cpu")
            
            self.is_loaded = True
            
            print(f"XTTS-v2 model loaded successfully")
            print(f"Using device: {'cuda' if self.tts.is_cuda_available else 'cpu'}")
            
            # Check language support
            # NOTE: Verify Arabic language code (usually "ar" or "ar-ar")
            print("NOTE: Verify Arabic language support in XTTS-v2 model")
            
        except ImportError as e:
            raise ImportError(
                f"Failed to import TTS: {e}\n"
                "Install with: pip install TTS"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load XTTS-v2 model: {e}")
    
    def synthesize(self, text: str, reference_audio_path: str, out_path: str) -> dict:
        """
        Synthesize speech using XTTS-v2 voice cloning.
        
        Args:
            text: Input text to synthesize
            reference_audio_path: Path to reference audio for voice cloning
            out_path: Path where output WAV file will be saved
        
        Returns:
            Dict with synthesis results
        """
        # Validate inputs
        self.validate_inputs(text, reference_audio_path, out_path)
        
        # Validate reference audio
        ref_validation = validate_audio(reference_audio_path)
        if not ref_validation["valid"]:
            return {
                "audio_path": out_path,
                "gen_time_sec": 0.0,
                "sample_rate": 16000,
                "success": False,
                "error": f"Invalid reference audio: {ref_validation.get('error', 'Unknown error')}"
            }
        
        if not self.is_loaded:
            return {
                "audio_path": out_path,
                "gen_time_sec": 0.0,
                "sample_rate": 16000,
                "success": False,
                "error": "Model not loaded. Call load() first."
            }
        
        try:
            start_time = time.time()
            
            # Synthesize using XTTS-v2
            # NOTE: The actual API call may differ - this is based on Coqui TTS API
            # Common pattern: tts.tts_to_file(text=text, speaker_wav=ref_audio, language="ar")
            self.tts.tts_to_file(
                text=text,
                speaker_wav=reference_audio_path,
                language="ar",  # Arabic language code - verify this
                file_path=out_path
            )
            
            gen_time = time.time() - start_time
            
            return {
                "audio_path": out_path,
                "gen_time_sec": gen_time,
                "sample_rate": 24000,  # XTTS-v2 typically outputs 24kHz
                "success": True,
                "error": None
            }
            
        except Exception as e:
            return {
                "audio_path": out_path,
                "gen_time_sec": 0.0,
                "sample_rate": 16000,
                "success": False,
                "error": f"Synthesis failed: {str(e)}"
            }


def main():
    parser = argparse.ArgumentParser(description="Run XTTS-v2 Arabic TTS pipeline")
    parser.add_argument("--ref", required=True, help="Path to reference audio file")
    parser.add_argument("--text", help="Text to synthesize (if not using eval sentences)")
    parser.add_argument("--out", help="Output path (default: results/samples/arabic/xtts/)")
    parser.add_argument("--config", help="Path to config file (default: configs/arabic.yaml)")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = XTTSArabicPipeline()
    
    # Load model
    print("Loading XTTS-v2 model...")
    pipeline.load(args.ref)
    
    # If single text provided
    if args.text:
        if not args.out:
            args.out = "results/samples/arabic/xtts/single.wav"
        
        result = pipeline.synthesize(args.text, args.ref, args.out)
        
        if result["success"]:
            print(f"Success! Audio saved to {result['audio_path']}")
            print(f"Generation time: {result['gen_time_sec']:.2f}s")
        else:
            print(f"Failed: {result['error']}")
    else:
        # Otherwise, load eval sentences and batch process
        eval_file = "data/eval_sentences/arabic.txt"
        if not os.path.exists(eval_file):
            print(f"Eval sentences file not found: {eval_file}")
            print("Please provide --text argument or create eval sentences file")
            return
        
        with open(eval_file, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        print(f"Processing {len(sentences)} sentences...")
        
        out_dir = "results/samples/arabic/xtts"
        os.makedirs(out_dir, exist_ok=True)
        
        results = []
        for i, sentence in enumerate(sentences):
            out_path = os.path.join(out_dir, f"sentence_{i:03d}.wav")
            print(f"[{i+1}/{len(sentences)}] {sentence[:50]}...")
            
            result = pipeline.synthesize(sentence, args.ref, out_path)
            results.append(result)
            
            if result["success"]:
                print(f"  -> Success ({result['gen_time_sec']:.2f}s)")
            else:
                print(f"  -> Failed: {result['error']}")
        
        # Summary
        successful = sum(1 for r in results if r["success"])
        print(f"\nCompleted: {successful}/{len(sentences)} successful")


if __name__ == "__main__":
    main()
