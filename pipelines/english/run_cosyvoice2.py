"""
English voice cloning pipeline using CosyVoice2 (FunAudioLLM).

CosyVoice2 supports 9 languages (Chinese, English, Japanese, Korean, German,
Spanish, French, Italian, Russian) - does NOT cover Arabic or Hindi, per its
own documentation. Used here for English only.
"""

import sys
import os
import time
import argparse
import torchaudio

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'CosyVoice'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'CosyVoice', 'third_party', 'Matcha-TTS'))

from tts_interface import TTSPipeline
from audio_utils import validate_audio


class CosyVoice2Pipeline(TTSPipeline):
    """
    CosyVoice2 voice cloning pipeline for English.
    """

    def __init__(self):
        super().__init__(model_name="cosyvoice2", language="en")
        self.model = None
        self.sample_rate = 24000  # updated after load from model.sample_rate
        self._prompt_text_cache = {}  # cache transcriptions per reference audio path

    def load(self, reference_audio_path: str = None) -> None:
        try:
            from cosyvoice.cli.cosyvoice import AutoModel

            model_dir = os.path.join(
                os.path.dirname(__file__), '..', '..', 'CosyVoice',
                'pretrained_models', 'CosyVoice2-0.5B'
            )
            self.model = AutoModel(model_dir=model_dir)
            self.sample_rate = self.model.sample_rate

            self.is_loaded = True
            print(f"CosyVoice2 model loaded successfully (sample_rate={self.sample_rate})")
        except ImportError as e:
            raise ImportError(f"Failed to import CosyVoice2: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load CosyVoice2 model: {e}")

    def _get_prompt_text(self, reference_audio_path: str) -> str:
        """
        Zero-shot cloning needs a transcription of the reference audio
        (prompt_text). Auto-generate it once with faster-whisper and cache it,
        rather than requiring the user to type it manually.
        """
        if reference_audio_path in self._prompt_text_cache:
            return self._prompt_text_cache[reference_audio_path]

        from faster_whisper import WhisperModel
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        whisper_model = WhisperModel("large-v3", device=device, compute_type=compute_type)

        segments, _ = whisper_model.transcribe(reference_audio_path, language="en", vad_filter=False)
        prompt_text = " ".join([s.text for s in segments]).strip()

        self._prompt_text_cache[reference_audio_path] = prompt_text
        print(f"Auto-transcribed reference audio for prompt_text: {prompt_text}")
        return prompt_text

    def synthesize(self, text: str, reference_audio_path: str, out_path: str) -> dict:
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
            out_dir = os.path.dirname(out_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

            prompt_text = self._get_prompt_text(reference_audio_path)

            start_time = time.time()

            # inference_zero_shot returns a generator; take the first (only,
            # since stream=False) chunk
            outputs = list(self.model.inference_zero_shot(
                text,
                prompt_text,
                reference_audio_path,
                stream=False
            ))

            gen_time = time.time() - start_time

            if not outputs:
                return {
                    "audio_path": out_path,
                    "gen_time_sec": gen_time,
                    "sample_rate": self.sample_rate,
                    "success": False,
                    "error": "inference_zero_shot returned no output"
                }

            wav = outputs[0]['tts_speech']
            torchaudio.save(out_path, wav, self.sample_rate)

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
    parser = argparse.ArgumentParser(description="Run CosyVoice2 English TTS pipeline")
    parser.add_argument("--ref", required=True, help="Path to reference audio file")
    parser.add_argument("--text", help="Text to synthesize (if not using eval sentences)")
    parser.add_argument("--out", help="Output path (default: results/samples/english/cosyvoice2/)")

    args = parser.parse_args()

    pipeline = CosyVoice2Pipeline()

    print("Loading CosyVoice2 model...")
    pipeline.load(args.ref)

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
        eval_file = "data/eval_sentences/english.txt"
        if not os.path.exists(eval_file):
            print(f"Eval sentences file not found: {eval_file}")
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

        successful = sum(1 for r in results if r["success"])
        print(f"\nCompleted: {successful}/{len(sentences)} successful")


if __name__ == "__main__":
    main()
