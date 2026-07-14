"""
English voice cloning pipeline using CosyVoice2 (FunAudioLLM).

CosyVoice2 is a streaming-capable TTS model with voice cloning support.
"""

import sys
import os
import time
import argparse

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))

from tts_interface import TTSPipeline
from audio_utils import load_audio, save_audio, validate_audio


class CosyVoice2Pipeline(TTSPipeline):
    """
    CosyVoice2 voice cloning pipeline for English.
    
    NOTE: CosyVoice2 is typically installed via git from FunAudioLLM/CosyVoice.
    The actual API may differ - this will need to be tested on Kaggle.
    """
    
    def __init__(self):
        super().__init__(model_name="cosyvoice2", language="en")
        self.model = None
    
    def load(self, reference_audio_path: str = None) -> None:
        """
        Load CosyVoice2 model.
        
        Args:
            reference_audio_path: Path to reference audio (may not be needed at load time)
        """
        try:
            # Import CosyVoice2
            # NOTE: The actual import path may differ - verify on Kaggle
            # CosyVoice is typically installed via: pip install git+https://github.com/FunAudioLLM/CosyVoice.git
            from cosyvoice.cli.cosyvoice import CosyVoice2
            
            # Initialize model
            # NOTE: Check actual initialization parameters in CosyVoice docs
            # Common pattern: CosyVoice2.load_model() or similar
            self.model = CosyVoice2.load_model()
            self.is_loaded = True
            
            print(f"CosyVoice2 model loaded successfully")
        except ImportError as e:
            raise ImportError(
                f"Failed to import CosyVoice2: {e}\n"
                "Install with: pip install git+https://github.com/FunAudioLLM/CosyVoice.git"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load CosyVoice2 model: {e}")
    
    def synthesize(self, text: str, reference_audio_path: str, out_path: str) -> dict:
        """
        Synthesize speech using CosyVoice2 voice cloning.
        
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
            
            # Load reference audio
            ref_audio, sr = load_audio(reference_audio_path, target_sr=16000)
            
            # Synthesize using CosyVoice2
            # NOTE: The actual API call may differ - this is a placeholder
            # Common pattern: model.inference(text, prompt_audio=ref_audio)
            generated_audio = self.model.inference(
                text=text,
                prompt_audio=ref_audio,
                prompt_text="",  # May need transcription of reference audio
                language="en"
            )
            
            gen_time = time.time() - start_time
            
            # Save output
            save_audio(generated_audio, out_path, sample_rate=16000)
            
            return {
                "audio_path": out_path,
                "gen_time_sec": gen_time,
                "sample_rate": 16000,
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
    parser = argparse.ArgumentParser(description="Run CosyVoice2 English TTS pipeline")
    parser.add_argument("--ref", required=True, help="Path to reference audio file")
    parser.add_argument("--text", help="Text to synthesize (if not using eval sentences)")
    parser.add_argument("--out", help="Output path (default: results/samples/english/cosyvoice2/)")
    parser.add_argument("--config", help="Path to config file (default: configs/english.yaml)")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = CosyVoice2Pipeline()
    
    # Load model
    print("Loading CosyVoice2 model...")
    pipeline.load(args.ref)
    
    # If single text provided
    if args.text:
        if not args.out:
            args.out = "results/samples/english/cosyvoice2/single.wav"
        
        result = pipeline.synthesize(args.text, args.ref, args.out)
        
        if result["success"]:
            print(f"Success! Audio saved to {result['audio_path']}")
            print(f"Generation time: {result['gen_time_sec']:.2f}s")
        else:
            print(f"Failed: {result['error']}")
    else:
        # Otherwise, load eval sentences and batch process
        eval_file = "data/eval_sentences/english.txt"
        if not os.path.exists(eval_file):
            print(f"Eval sentences file not found: {eval_file}")
            print("Please provide --text argument or create eval sentences file")
            return
        
        with open(eval_file, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        print(f"Processing {len(sentences)} sentences...")
        
        out_dir = "results/samples/english/cosyvoice2"
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
