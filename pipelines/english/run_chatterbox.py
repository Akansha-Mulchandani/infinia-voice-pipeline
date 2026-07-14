"""
English voice cloning pipeline using Chatterbox (Resemble AI).

Chatterbox is a zero-shot voice cloning model optimized for low latency.
"""

import sys
import os
import time
import argparse
import torch
import torchaudio as ta

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))

from tts_interface import TTSPipeline
from audio_utils import load_audio, save_audio, validate_audio


class ChatterboxPipeline(TTSPipeline):
    """
    Chatterbox voice cloning pipeline for English.
    """

    def __init__(self):
        super().__init__(model_name="chatterbox", language="en")
        self.model = None
        self.sample_rate = 24000  # chatterbox-tts default output SR; updated after load()

    def load(self, reference_audio_path: str = None) -> None:
        """
        Load Chatterbox model.

        Args:
            reference_audio_path: Path to reference audio (not needed at load time
                                   for chatterbox-tts; kept for interface compatibility)
        """
        try:
            from chatterbox.tts import ChatterboxTTS

            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Loading ChatterboxTTS on device: {device}")

            self.model = ChatterboxTTS.from_pretrained(device=device)

            # Use the model's actual native sample rate rather than assuming 16kHz
            self.sample_rate = getattr(self.model, "sr", 24000)

            self.is_loaded = True
            print(f"Chatterbox model loaded successfully (sample_rate={self.sample_rate})")
        except ImportError as e:
            raise ImportError(
                f"Failed to import chatterbox: {e}\n"
                "Install with: pip install chatterbox-tts"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load Chatterbox model: {e}")

    def synthesize(self, text: str, reference_audio_path: str, out_path: str) -> dict:
        """
        Synthesize speech using Chatterbox voice cloning.

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
                "sample_rate": self.sample_rate,
                "success": False,
                "error": f"Invalid reference audio: {ref_validation.get('error', 'Unknown error')}"
            }

        if not self.is_loaded:
            return {
                "audio_path": out_path,
                "gen_time_sec": 0.0,
                "sample_rate": self.sample_rate,
                "success": False,
                "error": "Model not loaded. Call load() first."
            }

        try:
            start_time = time.time()

            # chatterbox-tts takes the reference audio as a file path directly
            # (audio_prompt_path), not a pre-loaded array - it handles loading
            # and resampling internally.
            wav = self.model.generate(
                text,
                audio_prompt_path=reference_audio_path
            )

            gen_time = time.time() - start_time

            # model.generate() returns a torch tensor shaped [1, num_samples]
            # at self.model.sr - save with torchaudio, not the generic save_audio
            # helper, to avoid an incorrect sample rate assumption.
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            ta.save(out_path, wav.cpu() if wav.is_cuda else wav, self.sample_rate)

            return {
                "audio_path": out_path,
                "gen_time_sec": gen_time,
                "sample_rate": self.sample_rate,
                "success": True,
                "error": None
            }

        except Exception as e:
            return {
                "audio_path": out_path,
                "gen_time_sec": 0.0,
                "sample_rate": self.sample_rate,
                "success": False,
                "error": f"Synthesis failed: {str(e)}"
            }


def main():
    parser = argparse.ArgumentParser(description="Run Chatterbox English TTS pipeline")
    parser.add_argument("--ref", required=True, help="Path to reference audio file")
    parser.add_argument("--text", help="Text to synthesize (if not using eval sentences)")
    parser.add_argument("--out", help="Output path (default: results/samples/english/chatterbox/)")
    parser.add_argument("--config", help="Path to config file (default: configs/english.yaml)")

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = ChatterboxPipeline()

    # Load model
    print("Loading Chatterbox model...")
    pipeline.load(args.ref)

    # If single text provided
    if args.text:
        if not args.out:
            args.out = "results/samples/english/chatterbox/single.wav"

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

        out_dir = "results/samples/english/chatterbox"
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
