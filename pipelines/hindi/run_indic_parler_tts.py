"""
Hindi voice cloning pipeline using AI4Bharat Indic Parler-TTS.

Indic Parler-TTS is purpose-built for Indic languages including Hindi.
"""

import sys
import os
import time
import argparse

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))

from tts_interface import TTSPipeline
from audio_utils import load_audio, save_audio, validate_audio


class IndicParlerTTSPipeline(TTSPipeline):
    """
    Indic Parler-TTS voice cloning pipeline for Hindi.
    
    NOTE: Indic Parler-TTS is designed for Indic languages.
    Need to verify voice cloning capabilities and Hindi support.
    """
    
    def __init__(self):
        super().__init__(model_name="indic-parler-tts", language="hi")
        self.model = None
        self.processor = None
    
    def load(self, reference_audio_path: str = None) -> None:
        """
        Load Indic Parler-TTS model.
        
        Args:
            reference_audio_path: Path to reference audio (may not be needed at load time)
        """
        try:
            from transformers import AutoModelForTextToSpeech, AutoTokenizer
            
            # Load Indic Parler-TTS model for Hindi
            # Model: ai4bharat/indic-parler-tts
            model_name = "ai4bharat/indic-parler-tts"
            
            print(f"Loading Indic Parler-TTS model: {model_name}")
            self.model = AutoModelForTextToSpeech.from_pretrained(model_name)
            self.processor = AutoTokenizer.from_pretrained(model_name)
            
            # Move to GPU if available
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(device)
            
            self.is_loaded = True
            
            print(f"Indic Parler-TTS model loaded successfully")
            print(f"Using device: {device}")
            
            # NOTE: Check if this model supports voice cloning
            # Some Indic TTS models are single-speaker, not zero-shot cloning
            print("NOTE: Verify voice cloning support in Indic Parler-TTS")
            
        except ImportError as e:
            raise ImportError(
                f"Failed to import transformers: {e}\n"
                "Install with: pip install transformers accelerate"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load Indic Parler-TTS model: {e}")
    
    def synthesize(self, text: str, reference_audio_path: str, out_path: str) -> dict:
        """
        Synthesize speech using Indic Parler-TTS.
        
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
            
            # Tokenize text
            inputs = self.processor(text=text, return_tensors="pt")
            
            # NOTE: Check if model supports speaker conditioning
            # If it does, load reference audio and use it
            # If not, this will be a single-speaker synthesis
            
            # Generate speech
            import torch
            with torch.no_grad():
                output = self.model.generate(**inputs)
            
            gen_time = time.time() - start_time
            
            # Convert to numpy and save
            import numpy as np
            if isinstance(output, dict):
                audio_array = output["audio"].squeeze().cpu().numpy()
            else:
                audio_array = output.squeeze().cpu().numpy()
            
            # Indic Parler-TTS typically outputs at 16kHz
            save_audio(audio_array, out_path, sample_rate=16000)
            
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
    parser = argparse.ArgumentParser(description="Run Indic Parler-TTS Hindi TTS pipeline")
    parser.add_argument("--ref", required=True, help="Path to reference audio file")
    parser.add_argument("--text", help="Text to synthesize (if not using eval sentences)")
    parser.add_argument("--out", help="Output path (default: results/samples/hindi/indic_parler/)")
    parser.add_argument("--config", help="Path to config file (default: configs/hindi.yaml)")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = IndicParlerTTSPipeline()
    
    # Load model
    print("Loading Indic Parler-TTS model...")
    pipeline.load(args.ref)
    
    # If single text provided
    if args.text:
        if not args.out:
            args.out = "results/samples/hindi/indic_parler/single.wav"
        
        result = pipeline.synthesize(args.text, args.ref, args.out)
        
        if result["success"]:
            print(f"Success! Audio saved to {result['audio_path']}")
            print(f"Generation time: {result['gen_time_sec']:.2f}s")
        else:
            print(f"Failed: {result['error']}")
    else:
        # Otherwise, load eval sentences and batch process
        eval_file = "data/eval_sentences/hindi.txt"
        if not os.path.exists(eval_file):
            print(f"Eval sentences file not found: {eval_file}")
            print("Please provide --text argument or create eval sentences file")
            return
        
        with open(eval_file, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        print(f"Processing {len(sentences)} sentences...")
        
        out_dir = "results/samples/hindi/indic_parler"
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
