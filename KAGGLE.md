# Kaggle Execution Guide

This document provides exact, copy-pasteable instructions for running the infinia-voice-pipeline on Kaggle's free GPU notebooks.

## Prerequisites

1. **Create a Kaggle account** at https://www.kaggle.com
2. **Create a new notebook**:
   - Go to "Code" → "New Notebook"
   - Set **Accelerator = GPU T4 x2** (in the right sidebar under "Settings")
   - Turn **Internet ON** (in the right sidebar under "Settings")
3. **Upload your reference audio**:
   - Option A: Upload `data/reference_audio/me.wav` directly to the notebook's working directory
   - Option B: Create a Kaggle Dataset with your reference audio and mount it

## Kaggle Limits

- **Session limit**: 12 hours per session
- **GPU hours**: ~30 GPU-hours/week on free tier
- **Persistence**: Session state does NOT persist between sessions unless you save model checkpoints as a Kaggle Dataset
- **Recommendation**: Save model checkpoints as a Kaggle Dataset if you plan to re-run across multiple days

## Step-by-Step Execution

### Cell 1: Clone the repository

```bash
!git clone https://github.com/<YOUR_USERNAME>/infinia-voice-pipeline.git
%cd infinia-voice-pipeline
```

Replace `<YOUR_USERNAME>` with your GitHub username.

### Cell 2: Install dependencies

```bash
!pip install -r requirements.txt
```

**Note**: This may take 5-10 minutes. Do not interrupt.

### Cell 3: Install TTS (git install)

TTS (Coqui TTS fork) needs to be installed separately via git due to Python 3.12 compatibility:

```bash
!pip install git+https://github.com/coqui-ai/TTS.git
```

**Note**: This is required for XTTS-v2 pipelines (Arabic and Hindi). If this fails, XTTS-v2 will not work.

### Cell 4: Install CosyVoice (git install)

CosyVoice needs to be installed separately via git:

```bash
!pip install git+https://github.com/FunAudioLLM/CosyVoice.git
```

**Note**: If this fails, the CosyVoice2 pipeline will not work, but other pipelines (Chatterbox, XTTS-v2, etc.) should still function.

### Cell 5: Upload reference audio

If you haven't already uploaded your reference audio, do so now:

```bash
# Option A: If you uploaded directly to the notebook working directory
# (skip this cell if you already uploaded me.wav)

# Option B: If using a Kaggle Dataset
# First, add your dataset in the right sidebar under "Add data"
# Then:
!cp /kaggle/input/<YOUR_DATASET_NAME>/me.wav data/reference_audio/me.wav
```

Verify the file exists:

```bash
!ls -lh data/reference_audio/me.wav
```

### Cell 6: Run English pipelines

#### Cell 6a: Chatterbox (English)

```bash
!python pipelines/english/run_chatterbox.py --ref data/reference_audio/me.wav
```

**Expected output**: Should process 12 sentences from `data/eval_sentences/english.txt` and save audio to `results/samples/english/chatterbox/`.

**If this fails**: Paste the error output to debug. Common issues:
- Chatterbox import error: The API may have changed - check the actual import path
- CUDA out of memory: Try reducing batch size or using CPU

#### Cell 6b: CosyVoice2 (English)

```bash
!python pipelines/english/run_cosyvoice2.py --ref data/reference_audio/me.wav
```

**Expected output**: Should process 12 sentences and save audio to `results/samples/english/cosyvoice2/`.

**If this fails**: Paste the error output. CosyVoice installation issues are common - see Cell 3.

### Cell 7: Run Arabic pipelines

#### Cell 7a: XTTS-v2 (Arabic)

```bash
!python pipelines/arabic/run_xtts.py --ref data/reference_audio/me.wav
```

**Expected output**: Should process 12 Arabic sentences and save audio to `results/samples/arabic/xtts/`.

**If this fails**: Check if XTTS-v2 supports Arabic in the installed version. The language code may need to be adjusted.

#### Cell 7b: MMS-TTS (Arabic)

```bash
!python pipelines/arabic/run_mms_tts.py --ref data/reference_audio/me.wav
```

**Expected output**: Should process 12 Arabic sentences and save audio to `results/samples/arabic/mms_tts/`.

**Important note**: MMS-TTS does NOT support voice cloning. The output will use the model's pre-trained voice, not your reference speaker. Speaker similarity metrics will be meaningless for this model.

### Cell 8: Run Hindi pipelines

#### Cell 8a: Indic Parler-TTS (Hindi)

```bash
!python pipelines/hindi/run_indic_parler_tts.py --ref data/reference_audio/me.wav
```

**Expected output**: Should process 12 Hindi sentences and save audio to `results/samples/hindi/indic_parler/`.

**If this fails**: Check if Indic Parler-TTS supports voice cloning. Some Indic TTS models are single-speaker only.

#### Cell 8b: XTTS-v2 (Hindi)

```bash
!python pipelines/hindi/run_xtts_hindi.py --ref data/reference_audio/me.wav
```

**Expected output**: Should process 12 Hindi sentences and save audio to `results/samples/hindi/xtts/`.

**If this fails**: Check if XTTS-v2 supports Hindi in the installed version. The language code may need to be adjusted.

### Cell 9: Collect MOS ratings (optional but recommended)

Download the generated audio samples and listen to them locally, then rate them:

```bash
# First, zip the samples for download
!zip -r samples.zip results/samples/
```

Download `samples.zip` from the "Output" tab, extract locally, and rate each clip using the MOS collection script:

```bash
# After downloading and rating locally, upload the ratings CSV
# Then compute averages:
!python eval/mos_collect.py --compute results/mos_raw_en_chatterbox.csv
!python eval/mos_collect.py --compute results/mos_raw_en_cosyvoice2.csv
!python eval/mos_collect.py --compute results/mos_raw_ar_xtts.csv
!python eval/mos_collect.py --compute results/mos_raw_ar_mms_tts.csv
!python eval/mos_collect.py --compute results/mos_raw_hi_indic_parler.csv
!python eval/mos_collect.py --compute results/mos_raw_hi_xtts.csv
```

**Note**: You can also run the interactive MOS collection directly in Kaggle, but you won't be able to hear the audio easily. It's better to download clips and rate them locally.

### Cell 10: Run full evaluation

```bash
!python eval/run_all_evals.py --hardware "T4 x2" --ref data/reference_audio/me.wav
```

**Expected output**: Should run all 6 evaluation metrics (MOS, speaker similarity, latency, RTF, WER) for all pipelines and save results to `results/results.csv`.

**If this fails**: Check that all sample directories exist and contain audio files. The evaluation will skip pipelines that don't have samples.

### Cell 11: Download results

```bash
!zip -r /kaggle/working/results.zip results/
```

Download `results.zip` from the "Output" tab before the session ends.

## HuggingFace Token (if needed)

Some models may require a HuggingFace token for gated repositories:

1. Accept the model license on https://huggingface.co (e.g., for XTTS-v2 if it's gated)
2. Get your token from https://huggingface.co/settings/tokens
3. Add it as a Kaggle Secret:
   - In the notebook, go to "Add-ons" → "Secrets"
   - Add a secret named `HF_TOKEN` with your token value
4. Use it in your code:

```python
from huggingface_hub import login
import os

token = os.environ.get("HF_TOKEN")
if token:
    login(token=token)
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**:
   - Reduce batch size in pipeline scripts
   - Use CPU instead (slower but works)
   - Close other notebooks to free GPU memory

2. **Import errors**:
   - The APIs for Chatterbox, CosyVoice, and Indic Parler-TTS may have changed
   - Paste the exact error message to debug
   - Check the actual import paths in the model documentation

3. **Language not supported**:
   - XTTS-v2 language support varies by version
   - Check the model's supported languages in its documentation
   - Adjust the language code in the pipeline script if needed

4. **Reference audio not found**:
   - Ensure `data/reference_audio/me.wav` exists
   - Check the file path is correct

5. **Session timeout**:
   - Kaggle sessions timeout after 12 hours
   - Save model checkpoints as a Kaggle Dataset to resume later
   - Download results before the session ends

### Debugging Failed Pipelines

If a pipeline fails:

1. **Paste the exact error output** - don't guess
2. **Check the error message** - it will tell you what went wrong
3. **Re-run only the failing cell** - don't re-run the whole notebook
4. **Check model documentation** - APIs change frequently

## Optimization Tips

1. **Batch processing**: All pipeline scripts batch-process eval sentences to minimize model loading overhead
2. **Model caching**: Save model checkpoints as Kaggle Datasets to avoid re-downloading
3. **Selective evaluation**: If you're debugging, run only the pipeline you're working on
4. **GPU utilization**: Monitor GPU usage in the right sidebar to ensure you're using the GPU

## After Evaluation

1. Download `results.zip` from the "Output" tab
2. Extract locally to view `results/results.csv`
3. Review the results and update the README.md with your findings
4. Commit and push changes to GitHub

## Next Steps

After running the evaluation on Kaggle:

1. Update `README.md` with your actual results from `results/results.csv`
2. Update `notes/model_comparison.md` with your findings
3. Update `notes/ambiguities.md` with any assumptions you made
4. Commit and push to GitHub
