"""
Arabic voice cloning pipeline using XTTS-v2 (Coqui).

XTTS-v2 is a multilingual zero-shot voice cloning model.
License note: XTTS-v2 is used here under Coqui's non-commercial CPML license
(https://coqui.ai/cpml) - not a fully open commercial license. Flagged here
for reproducibility/licensing transparency per the brief's requirements.
"""

import sys
import os
import time
import argparse

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))

from tts_interface import TTSPipeline
from audio_utils import validate_audio


class XTTSArabicPipeline(TTSPipeline):
    """
    XTTS-v2 voice cloning pipeline for Arabic (Modern Standard Arabic).
    """

    def __init__(self):
        super().__init__(model_name="xtts-v2", language="ar")
        self.model = None
        self.sample_rate = 24000  # XTTS-v2 native output rate

    def load(self, reference_audio_path: str = None) -> None:
        """
        Load XTTS-v2 model.

        Args:
            reference_audio_path: Path to reference audio (not needed at load
                                   time for XTTS-v2; kept for interface compatibility)
        """
        try:
            from TTS.api import TTS

            self.model = TTS("tts_models/multilingual/multi-dataset/xtts_v2")

            # Confirm Arabic is actually supported by this installed version
            # before proceeding - fail loudly and clearly if not, rather than
            # letting a downstream synthesis call throw an unclear error.
            if hasattr(self.model, "languages") and "ar" not in self.model.languages:
                raise RuntimeError(
                    f"Arabic ('ar') not in this XTTS-v2 build's supported "
                    f"languages: {self.model.languages}"
                )

            self.is_loaded = True
            print(f"XTTS-v2 model loaded successfully (Arabic supported)")
        except ImportError as e:
            raise ImportError(
                f"Failed to import TTS: {e}\n"
                "Install with: pip install coqui-tts"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load XTTS-v2 model: {e}")

    def synthesize(self, text: str, reference_audio_path: str, out_path: str) -> dict:
        """
        Synthesize Arabic speech using XTTS-v2 voice cloning.

        Args:
            text: Input Arabic text to synthesize
            reference_audio_path: Path to reference audio for voice cloning
            out_path: Path where output WAV file will be saved

        Returns:
            Dict with synthesis results
        """
        self.validate_inputs(text, reference_audio_path, out_path)

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
            # Guard against bare filenames with no directory component
            # (same fix applied across all pipelines after the eval harness bug)
            out_dir = os.path.dirname(out_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

            start_time = time.time()

            # tts_to_file writes directly to disk and handles sample rate
            # internally - no manual save step needed, unlike Chatterbox.
            self.model.tts_to_file(
                text=text,
                speaker_wav=reference_audio_path,
                language="ar",
                file_path=out_path
            )

            gen_time = time.time() - start_time

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
    parser = argparse.ArgumentParser(description="Run XTTS-v2 Arabic TTS pipeline")
    parser.add_argument("--ref", required=True, help="Path to reference audio file")
    parser.add_argument("--text", help="Text to synthesize (if not using eval sentences)")
    parser.add_argument("--out", help="Output path (default: results/samples/arabic/xtts/)")

    args = parser.parse_args()

    pipeline = XTTSArabicPipeline()

    print("Loading XTTS-v2 model...")
    pipeline.load(args.ref)

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

        successful = sum(1 for r in results if r["success"])
        print(f"\nCompleted: {successful}/{len(sentences)} successful")


if __name__ == "__main__":
    main()
