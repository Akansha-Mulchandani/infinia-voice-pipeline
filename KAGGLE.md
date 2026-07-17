# Kaggle Execution Guide

This project was built and run entirely on Kaggle's free GPU notebooks (no local GPU was available). Several of the six models tested here have conflicting dependency requirements, so they were run in six separate Kaggle notebook sessions rather than one shared environment. This doc gives the real, working commands for each, in the order they were actually run.

## Prerequisites (every session)

1. Kaggle account at kaggle.com
2. New notebook: Code -> New Notebook
3. Right sidebar -> Settings -> Accelerator: **GPU T4 x2**
4. Right sidebar -> Settings -> Internet: **On**
5. Add your reference audio as a Kaggle Dataset once (Add Input in the right sidebar), then reuse it across every session below with:
   ```bash
   !mkdir -p data/reference_audio
   !find /kaggle/input -name "mef.wav"
   # copy using whatever path the above prints, e.g.:
   !cp /kaggle/input/<your-dataset-name>/mef.wav data/reference_audio/mef.wav
   !ls -la data/reference_audio/
   ```

## Kaggle limits to plan around

- 12 hour session limit
- Roughly 30 GPU-hours/week on the free tier
- Nothing persists between sessions except what you download or save as a Dataset
- Zip and download your results before a session ends, every time

## Session 1: English, Chatterbox

```bash
!git clone https://github.com/Akansha-Mulchandani/infinia-voice-pipeline.git
%cd infinia-voice-pipeline
!pip install -r requirements.txt
!pip install coqui-tts
```
(reference audio setup as above)
```bash
!python pipelines/english/run_chatterbox.py --ref data/reference_audio/mef.wav
!pip install resemblyzer faster-whisper jiwer
!python eval/run_all_evals.py --hardware "T4 x2" --ref data/reference_audio/mef.wav
```

**Known fix needed in this session:** `chatterbox-tts` pulls in `transformers 5.x`, which conflicts with `torchvision`. If you hit `RuntimeError: operator torchvision::nms does not exist`, run:
```bash
!pip uninstall -y torchvision
```
then restart the kernel and re-run.

**If faster-whisper throws `RecursionError: maximum recursion depth exceeded`** during WER scoring, the fix is disabling VAD filtering (already applied in `eval/round_trip_wer.py` in this repo, `vad_filter=False`). If you still see it, it's usually a corrupted numpy install picked up along the way:
```bash
!pip install --force-reinstall --no-deps numpy==1.26.4
```
then restart the kernel.

```bash
!zip -r /kaggle/working/english_results.zip results/ data/reference_audio/
```
Download from the Output tab.

## Session 2: Arabic, XTTS-v2

```bash
!git clone https://github.com/Akansha-Mulchandani/infinia-voice-pipeline.git
%cd infinia-voice-pipeline
!pip install coqui-tts resemblyzer faster-whisper jiwer
```
(reference audio setup as above)
```bash
!python pipelines/arabic/run_xtts.py --ref data/reference_audio/mef.wav
```
This will prompt you to accept Coqui's non-commercial CPML license (y/n) on first model load. If the interactive prompt doesn't respond to typing in the notebook UI, pre-accept it instead:
```python
import os
os.environ["COQUI_TOS_AGREED"] = "1"
```
run that before the pipeline script, in a separate cell.

```bash
!python eval/run_all_evals.py --hardware "T4 x2" --ref data/reference_audio/mef.wav
!zip -r /kaggle/working/arabic_full_results.zip results/samples/arabic/ results/results.csv data/eval_sentences/arabic.txt
```
Download from the Output tab.

## Session 3: Hindi, XTTS-v2

Same setup as Session 2 (a separate session since XTTS-v2's dependencies conflict with Chatterbox's):
```bash
!git clone https://github.com/Akansha-Mulchandani/infinia-voice-pipeline.git
%cd infinia-voice-pipeline
!pip install coqui-tts resemblyzer faster-whisper jiwer
```
(reference audio setup as above)
```bash
!python pipelines/hindi/run_xtts_hindi.py --ref data/reference_audio/mef.wav
```
**Known issue, not a bug to fix:** 2 of 12 Hindi sentences fail with an empty `NotImplementedError`. Root cause is that XTTS-v2's Hindi number expansion depends on `num2words`, which has no Hindi converter. Both failures are on sentences containing digits. See notes/model_comparison.md for detail. This is expected in this repo's current state.

```bash
!python eval/run_all_evals.py --hardware "T4 x2" --ref data/reference_audio/mef.wav
!zip -r /kaggle/working/hindi_full_results.zip results/samples/hindi/xtts/ results/results.csv
```
Download from the Output tab.

## Session 4: Arabic, MMS-TTS

```bash
!git clone https://github.com/Akansha-Mulchandani/infinia-voice-pipeline.git
%cd infinia-voice-pipeline
!pip install transformers accelerate resemblyzer faster-whisper jiwer
```
(reference audio setup as above, note: MMS-TTS does not actually use the reference audio, it does not support voice cloning, `--ref` is kept only for script argument consistency)
```bash
!python pipelines/arabic/run_mms_tts.py --ref data/reference_audio/mef.wav
!python eval/run_all_evals.py --hardware "T4 x2" --ref data/reference_audio/mef.wav
!zip -r /kaggle/working/mms_tts_full_results.zip results/samples/arabic/mms_tts/ results/results.csv results/mos_raw_ar_mms-tts.csv
```
Download from the Output tab. This session ran the cleanest of all six, no dependency issues encountered.

## Session 5: Hindi, Indic Parler-TTS

```bash
!git clone https://github.com/Akansha-Mulchandani/infinia-voice-pipeline.git
%cd infinia-voice-pipeline
!pip install git+https://github.com/huggingface/parler-tts.git transformers accelerate resemblyzer faster-whisper jiwer
```
(reference audio setup as above, note: this model also does not clone voices, it is description-conditioned, see notes/model_comparison.md)
```bash
!python pipelines/hindi/run_indic_parler_tts.py --ref data/reference_audio/mef.wav
```
**Known fix needed in this session:** an outdated `protobuf` causes `ImportError: cannot import name 'runtime_version' from 'google.protobuf'` when `transformers` tries to import `tensorflow` internally. Fix:
```bash
!pip install --upgrade protobuf
```
then restart the kernel and re-run.

```bash
!python eval/run_all_evals.py --hardware "T4 x2" --ref data/reference_audio/mef.wav
!zip -r /kaggle/working/indic_parler_tts_full_results.zip results/samples/hindi/indic_parler/ results/results.csv results/mos_raw_hi_indic-parler-tts.csv
```
Download from the Output tab.

## Session 6: English, CosyVoice2

This is the hardest session by far, eight separate dependency issues were hit and fixed here across two actual attempts. Full detail on all of them is in notes/model_comparison.md. Commands below are the real, working sequence, not a guess.

```bash
!git clone https://github.com/Akansha-Mulchandani/infinia-voice-pipeline.git
%cd infinia-voice-pipeline
!git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
```

Install CosyVoice's dependencies, skipping known-broken pinned packages (`openai-whisper` old pin, `deepspeed`, `tensorrt-cu12*`, old `grpcio` pins):
```bash
!grep -v -E "^(openai-whisper|deepspeed|tensorrt-cu12|grpcio==|grpcio-tools==)" CosyVoice/requirements.txt > CosyVoice/requirements_filtered.txt
!pip install -r CosyVoice/requirements_filtered.txt
!pip install grpcio grpcio-tools
!pip install -U openai-whisper
!pip uninstall -y torchvision
!pip install setuptools==68.0.0
```

Restart the kernel after the above (mandatory, several of these fixes require a clean Python state). Then:
```bash
%cd /kaggle/working/infinia-voice-pipeline
```

Download the CosyVoice2 model weights:
```python
from huggingface_hub import snapshot_download
snapshot_download('FunAudioLLM/CosyVoice2-0.5B', local_dir='CosyVoice/pretrained_models/CosyVoice2-0.5B')
```

(reference audio setup as above)
```bash
!pip install faster-whisper resemblyzer jiwer
!python pipelines/english/run_cosyvoice2.py --ref data/reference_audio/mef.wav
```

**Known limitation:** `onnxruntime-gpu` did not get correctly installed across the dependency juggling above, so CosyVoice2 ran on CPU rather than GPU in this project (visible as `Specified provider 'CUDAExecutionProvider' is not in available provider names` in the logs). Its latency and RTF numbers are disclosed as CPU-run in results.csv and are not directly comparable to the other, GPU-run models. Not resolved here; the likely fix is a clean reinstall of `onnxruntime-gpu` matching the CUDA version, attempted separately from the rest of this dependency chain.

**Also known:** reference audio over 30 seconds will fail with `AssertionError: do not support extract speech token for audio longer than 30s`. This pipeline auto-trims to 25 seconds internally, so no manual action is needed, but it's worth knowing if you see it while experimenting directly with the model outside this repo's scripts.

```bash
!python eval/run_all_evals.py --hardware "T4 x2 (CosyVoice2 ran CPU-only, onnxruntime-gpu not installed)" --ref data/reference_audio/mef.wav
!zip -r /kaggle/working/cosyvoice2_full_results.zip results/samples/english/cosyvoice2/ results/results.csv results/mos_raw_en_cosyvoice2.csv
```
Download from the Output tab.

## After all six sessions

1. Download all six results zips (one per session above).
2. Unzip each locally.
3. Merge the six results.csv files into one final results/results.csv (this repo's final version already reflects that merge).
4. Upload the generated audio samples from all six unzipped folders into this repo under results/samples/, matching the existing folder structure.
5. MOS ratings need a human listener. For each model, download its samples, listen to at least a few clips, and log ratings to a CSV named results/mos_raw_{language}_{model}.csv with columns sample_file, rater, rating (the hyphen/underscore in the model name must exactly match the model's registered name, this project hit a real bug from a filename mismatch here, see notes/model_comparison.md).

## HuggingFace token (only needed if a model is gated)

Not needed for any of the six models in this project as run. If a future model requires it:
1. Accept the model's license on huggingface.co
2. Get a token from huggingface.co/settings/tokens
3. Add it as a Kaggle Secret (Add-ons -> Secrets)
4. In the notebook:
```python
from huggingface_hub import login
import os
login(token=os.environ.get("HF_TOKEN"))
```

## General troubleshooting notes from this project

- **Silent numpy corruption** was the single most recurring issue across every session, showing up as anything from `RecursionError` to `ImportError: numpy.dtype size changed`. Whenever something breaks in a way that doesn't match the error you'd expect, try `pip install --force-reinstall --no-deps numpy==1.26.4` followed by a full kernel restart before assuming the real cause is something else.
- **Restart the kernel after any dependency fix**, not just after the specific error you were chasing. Python caches failed imports, so a fix can appear to not work simply because the broken import is still cached in memory.
- **Check your working directory** after every restart with `!pwd`. Kernel restarts do not change what's on disk, but they do reset Python's current directory, and relative paths (`./CosyVoice`, `data/reference_audio/mef.wav`) will silently fail if you're not back in the repo root.
