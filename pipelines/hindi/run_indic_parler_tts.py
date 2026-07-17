"""
Hindi TTS pipeline using AI4Bharat Indic Parler-TTS.

IMPORTANT LIMITATION: Indic Parler-TTS is a description-conditioned TTS model,
NOT a zero-shot voice cloning model. Instead of cloning a reference audio, you
describe the desired voice characteristics in a text prompt (e.g. "a female
speaker with a clear voice"). It does NOT use the reference audio file at all.
This is a different limitation than MMS-TTS (which is fixed-single-speaker),
but has the same practical effect: it cannot reproduce the reference speaker's
actual voice. This is a documented finding, not a bug.
"""

import sys
import os
import time
import argparse

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))

from tts_interface import TTSPipeline
from audio_utils import save_audio


class IndicParlerTTSPipeline(TTSPipeline):
    """
    Indic Parler-TTS pipeline for Hindi.

    NOT a voice-cloning model - see module docstring. reference_audio_path is
    accepted for interface compatibility but is NOT used in synthesis.
    """

    def __init__(self):
        super().__init__(model_name="indic-parler-tts", language="hi")
        self.model = None
        self.tokenizer = None
        self.description_tokenizer = None
        self.sample_rate = 16000
        # Fixed voice description - Parler-TTS uses this instead of cloning
        self.voice_description = (
            "A clear, natural-sounding Hindi speaker talks at a moderate pace "
            "with good audio quality and minimal background noise."
        )

    def load(self, reference_audio_path: str = None) -> None:
        try:
            from parler_tts import ParlerTTSForConditionalGeneration
            from transformers import AutoTokenizer
            import torch

            model_name = "ai4bharat/indic-parler-tts"
            print(f"Loading Indic Parler-TTS model: {model_name}")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.description_tokenizer = AutoTokenizer.from_pretrained(
                self.model.config.text_encoder._name_or_path
            )
            self.sample_rate = self.model.config.sampling_rate

            self.is_loaded = True
            print(f"Indic Parler-TTS model loaded successfully (device={device}, sample_rate={self.sample_rate})")
            print("IMPORTANT: Indic Parler-TTS does NOT support voice cloning.")
            print("It uses a fixed text description of voice characteristics instead.")

        except ImportError as e:
            raise ImportError(
                f"Failed to import parler_tts: {e}\n"
                "Install with: pip install git+https://github.com/huggingface/parler-tts.git"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load Indic Parler-TTS model: {e}")

    def synthesize(self, text: str, reference_audio_path: str, out_path: str) -> dict:
        if not text or not text.strip():
            return {
                "audio_path": out_path, "gen_time_sec": 0.0, "sample_rate": self.sample_rate,
                "success": False, "error": "Text cannot be empty"
            }

        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        if not self.is_loaded:
            return {
                "audio_path": out_path, "gen_time_sec": 0.0, "sample_rate": self.sample_rate,
                "success": False, "error": "Model not loaded. Call load() first."
            }

        try:
            import torch

            device = next(self.model.parameters()).device
            start_time = time.time()

            description_ids = self.description_tokenizer(
                self.voice_description, return_tensors="pt"
            ).input_ids.to(device)
            prompt_ids = self.tokenizer(text, return_tensors="pt").input_ids.to(device)

            with torch.no_grad():
                generation = self.model.generate(
                    input_ids=description_ids,
                    prompt_input_ids=prompt_ids
                )

            gen_time = time.time() - start_time

            audio_array = generation.cpu().numpy().squeeze()
            save_audio(audio_array, out_path, sample_rate=self.sample_rate)

            return {
                "audio_path": out_path,
                "gen_time_sec": gen_time,
                "sample_rate": self.sample_rate,
                "success": True,
                "error": None,
                "note": "Indic Parler-TTS does not support voice cloning - uses fixed voice description"
            }

        except Exception as e:
            return {
                "audio_path": out_path, "gen_time_sec": 0.0, "sample_rate": self.sample_rate,
                "success": False, "error": f"Synthesis failed: {str(e)}"
            }


def main():
    parser = argparse.ArgumentParser(description="Run Indic Parler-TTS Hindi TTS pipeline")
    parser.add_argument("--ref", required=True, help="Path to reference audio (NOT USED - not a cloning model)")
    parser.add_argument("--text", help="Text to synthesize (if not using eval sentences)")
    parser.add_argument("--out", help="Output path (default: results/samples/hindi/indic_parler/)")
    args = parser.parse_args()

    pipeline = IndicParlerTTSPipeline()
    print("Loading Indic Parler-TTS model...")
    pipeline.load(args.ref)

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
        eval_file = "data/eval_sentences/hindi.txt"
        if not os.path.exists(eval_file):
            print(f"Eval sentences file not found: {eval_file}")
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
            print(f"  -> Success ({result['gen_time_sec']:.2f}s)" if result["success"] else f"  -> Failed: {result['error']}")

        successful = sum(1 for r in results if r["success"])
        print(f"\nCompleted: {successful}/{len(sentences)} successful")
        print("IMPORTANT: Indic Parler-TTS does not support voice cloning.")
        print("Speaker similarity metrics will be meaningless for this model.")


if __name__ == "__main__":
    main()
