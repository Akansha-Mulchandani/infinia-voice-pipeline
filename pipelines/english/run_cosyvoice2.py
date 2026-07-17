"""
English voice cloning pipeline using CosyVoice2 (FunAudioLLM).

CosyVoice2 supports 9 languages (Chinese, English, Japanese, Korean, German,
Spanish, French, Italian, Russian) - does NOT cover Arabic or Hindi.
NOTE: CosyVoice2 has a hard 30-second cap on reference/prompt audio length
for zero-shot cloning - reference audio is auto-trimmed to 25s if longer.
"""

import sys
import os
import time
import argparse
import torchaudio
import soundfile as sf

sys.path.insert(0, '/kaggle/working/infinia-voice-pipeline/CosyVoice')
sys.path.insert(0, '/kaggle/working/infinia-voice-pipeline/CosyVoice/third_party/Matcha-TTS')
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))

from tts_interface import TTSPipeline
from audio_utils import validate_audio


class CosyVoice2Pipeline(TTSPipeline):
    def __init__(self):
        super().__init__(model_name="cosyvoice2", language="en")
        self.model = None
        self.sample_rate = 24000
        self._prompt_text_cache = {}

    def _get_trimmed_ref(self, reference_audio_path: str) -> str:
        """CosyVoice2 rejects reference audio longer than 30s - trim to 25s if needed."""
        audio, sr = sf.read(reference_audio_path)
        duration = len(audio) / sr
        if duration <= 28:
            return reference_audio_path

        trimmed_path = reference_audio_path.replace('.wav', '_trimmed.wav')
        max_samples = int(25 * sr)
        sf.write(trimmed_path, audio[:max_samples], sr)
        print(f"Reference audio was {duration:.1f}s, trimmed to 25s at {trimmed_path}")
        return trimmed_path

    def load(self, reference_audio_path: str = None) -> None:
        try:
            from cosyvoice.cli.cosyvoice import AutoModel

            model_dir = '/kaggle/working/infinia-voice-pipeline/CosyVoice/pretrained_models/CosyVoice2-0.5B'
            self.model = AutoModel(model_dir=model_dir)
            self.sample_rate = self.model.sample_rate

            self.is_loaded = True
            print(f"CosyVoice2 model loaded successfully (sample_rate={self.sample_rate})")
        except ImportError as e:
            raise ImportError(f"Failed to import CosyVoice2: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load CosyVoice2 model: {e}")

    def _get_prompt_text(self, reference_audio_path: str) -> str:
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
        print(f"Auto-transcribed prompt_text: {prompt_text}")
        return prompt_text

    def synthesize(self, text: str, reference_audio_path: str, out_path: str) -> dict:
        self.validate_inputs(text, reference_audio_path, out_path)

        ref_validation = validate_audio(reference_audio_path)
        if not ref_validation["valid"]:
            return {
                "audio_path": out_path, "gen_time_sec": 0.0, "sample_rate": self.sample_rate,
                "success": False, "error": f"Invalid reference audio: {ref_validation.get('error')}"
            }

        if not self.is_loaded:
            return {
                "audio_path": out_path, "gen_time_sec": 0.0, "sample_rate": self.sample_rate,
                "success": False, "error": "Model not loaded. Call load() first."
            }

        try:
            out_dir = os.path.dirname(out_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

            trimmed_ref = self._get_trimmed_ref(reference_audio_path)
            prompt_text = self._get_prompt_text(trimmed_ref)

            start_time = time.time()
            outputs = list(self.model.inference_zero_shot(text, prompt_text, trimmed_ref, stream=False))
            gen_time = time.time() - start_time

            if not outputs:
                return {
                    "audio_path": out_path, "gen_time_sec": gen_time, "sample_rate": self.sample_rate,
                    "success": False, "error": "inference_zero_shot returned no output"
                }

            wav = outputs[0]['tts_speech']
            torchaudio.save(out_path, wav, self.sample_rate)

            return {
                "audio_path": out_path, "gen_time_sec": gen_time, "sample_rate": self.sample_rate,
                "success": True, "error": None
            }
        except Exception as e:
            return {
                "audio_path": out_path, "gen_time_sec": 0.0, "sample_rate": self.sample_rate,
                "success": False, "error": f"Synthesis failed: {str(e)}"
            }


def main():
    parser = argparse.ArgumentParser(description="Run CosyVoice2 English TTS pipeline")
    parser.add_argument("--ref", required=True)
    parser.add_argument("--text")
    parser.add_argument("--out")
    args = parser.parse_args()

    pipeline = CosyVoice2Pipeline()
    print("Loading CosyVoice2 model...")
    pipeline.load(args.ref)

    if args.text:
        if not args.out:
            args.out = "results/samples/english/cosyvoice2/single.wav"
        result = pipeline.synthesize(args.text, args.ref, args.out)
        print(f"Success! {result}" if result["success"] else f"Failed: {result['error']}")
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
            print(f"  -> Success ({result['gen_time_sec']:.2f}s)" if result["success"] else f"  -> Failed: {result['error']}")

        successful = sum(1 for r in results if r["success"])
        print(f"\nCompleted: {successful}/{len(sentences)} successful")


if __name__ == "__main__":
    main()
