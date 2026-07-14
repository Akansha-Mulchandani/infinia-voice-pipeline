"""
Arabic TTS pipeline using Meta MMS-TTS (Massively Multilingual Speech).

MMS-TTS provides broad language coverage but typically does NOT support
zero-shot voice cloning - it's a fixed single-speaker model per language.

This is a documented limitation, not a bug.
"""

import sys
import os
import time
import argparse

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))

from tts_interface import TTSPipeline
from audio_utils import load_audio, save_audio, validate_audio


class MMSArabicPipeline(TTSPipeline):
    """
    MMS-TTS pipeline for Arabic.
    
    IMPORTANT LIMITATION: MMS-TTS is typically a fixed single-speaker model
    per language and does NOT support zero-shot voice cloning from reference audio.
    
    This pipeline will synthesize Arabic speech but will NOT clone the reference
    speaker's voice. This is a fundamental limitation of the MMS-TTS architecture.
    """
    
    def __init__(self):
        super().__init__(model_name="mms-tts", language="ar")
        self.model = None
        self.processor = None
    
    def load(self, reference_audio_path: str = None) -> None:
        """
        Load MMS-TTS model.
        
        Args:
            reference_audio_path: Not used (MMS-TTS doesn't support voice cloning)
        """
        try:
            from transformers import VitsModel, AutoTokenizer
            
            # Load MMS-TTS model for Arabic
            # Model: facebook/mms-tts-ara
            model_name = "facebook/mms-tts-ara"
            
            print(f"Loading MMS-TTS model: {model_name}")
            self.model = VitsModel.from_pretrained(model_name)
            self.processor = AutoTokenizer.from_pretrained(model_name)
            
            # Move to GPU if available
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model.to(device)
            
            self.is_loaded = True
            
            print(f"MMS-TTS model loaded successfully")
            print(f"Using device: {device}")
            print("IMPORTANT: MMS-TTS does NOT support voice cloning.")
            print("Output will use the model's pre-trained voice, not the reference speaker.")
            
        except ImportError as e:
            raise ImportError(
                f"Failed to import transformers: {e}\n"
                "Install with: pip install transformers accelerate"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load MMS-TTS model: {e}")
    
    def synthesize(self, text: str, reference_audio_path: str, out_path: str) -> dict:
        """
        Synthesize speech using MMS-TTS.
        
        NOTE: This will NOT clone the reference speaker's voice.
        MMS-TTS uses a fixed pre-trained voice for each language.
        
        Args:
            text: Input text to synthesize
            reference_audio_path: Path to reference audio (NOT USED - MMS-TTS doesn't support cloning)
            out_path: Path where output WAV file will be saved
        
        Returns:
            Dict with synthesis results
        """
        # Validate inputs (but note reference_audio is not used)
        if not text or not text.strip():
            return {
                "audio_path": out_path,
                "gen_time_sec": 0.0,
                "sample_rate": 16000,
                "success": False,
                "error": "Text cannot be empty"
            }
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
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
            
            # Generate speech
            import torch
            with torch.no_grad():
                output = self.model(**inputs).waveform
            
            gen_time = time.time() - start_time
            
            # Convert to numpy and save
            import numpy as np
            audio_array = output.squeeze().cpu().numpy()
            
            # MMS-TTS typically outputs at 16kHz
            save_audio(audio_array, out_path, sample_rate=16000)
            
            return {
                "audio_path": out_path,
                "gen_time_sec": gen_time,
                "sample_rate": 16000,
                "success": True,
                "error": None,
                "note": "MMS-TTS does not support voice cloning - output uses pre-trained voice"
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
    parser = argparse.ArgumentParser(description="Run MMS-TTS Arabic TTS pipeline")
    parser.add_argument("--ref", help="Path to reference audio file (NOT USED - MMS-TTS doesn't support cloning)")
    parser.add_argument("--text", help="Text to synthesize (if not using eval sentences)")
    parser.add_argument("--out", help="Output path (default: results/samples/arabic/mms_tts/)")
    parser.add_argument("--config", help="Path to config file (default: configs/arabic.yaml)")
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = MMSArabicPipeline()
    
    # Load model
    print("Loading MMS-TTS model...")
    pipeline.load(args.ref)
    
    # If single text provided
    if args.text:
        if not args.out:
            args.out = "results/samples/arabic/mms_tts/single.wav"
        
        result = pipeline.synthesize(args.text, args.ref or "", args.out)
        
        if result["success"]:
            print(f"Success! Audio saved to {result['audio_path']}")
            print(f"Generation time: {result['gen_time_sec']:.2f}s")
            if "note" in result:
                print(f"Note: {result['note']}")
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
        
        out_dir = "results/samples/arabic/mms_tts"
        os.makedirs(out_dir, exist_ok=True)
        
        results = []
        for i, sentence in enumerate(sentences):
            out_path = os.path.join(out_dir, f"sentence_{i:03d}.wav")
            print(f"[{i+1}/{len(sentences)}] {sentence[:50]}...")
            
            result = pipeline.synthesize(sentence, args.ref or "", out_path)
            results.append(result)
            
            if result["success"]:
                print(f"  -> Success ({result['gen_time_sec']:.2f}s)")
            else:
                print(f"  -> Failed: {result['error']}")
        
        # Summary
        successful = sum(1 for r in results if r["success"])
        print(f"\nCompleted: {successful}/{len(sentences)} successful")
        print("IMPORTANT: MMS-TTS does not support voice cloning.")
        print("Speaker similarity metrics will be meaningless for this model.")


if __name__ == "__main__":
    main()
