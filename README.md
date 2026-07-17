# infinia-voice-pipeline

Open-source voice cloning TTS pipelines for English, Arabic (MSA), and Hindi, with a shared evaluation harness measuring six metrics (MOS, speaker similarity, latency, RTF, WER) across six models. Recommended setup: Chatterbox for English, XTTS-v2 for Arabic and Hindi (with MMS-TTS as a fast, non-cloning fallback for Arabic). Full reasoning and every issue hit along the way is in notes/model_comparison.md, this README covers the numbers and setup.

## Setup and reproduction

### Prerequisites

- Python 3.10+ (tested on Python 3.12, Kaggle's default)
- GPU with 16GB VRAM (tested on Kaggle T4 x2, free tier)
- No local NVIDIA GPU is required; this project was built and run entirely on Kaggle's free GPU notebooks
- Reference audio: 15 to 30 seconds of clean mono WAV at 16kHz or higher (see notes/ambiguities.md for what was actually used)

### Kaggle execution (how this was actually run)

See **KAGGLE.md** for exact step-by-step notebook instructions. Short version: clone this repo into a Kaggle notebook with GPU enabled, install the requirements for whichever model you want to run, add your reference audio, then run the relevant script in `pipelines/`.

Note: several models in this project have conflicting dependency requirements and were run in separate Kaggle sessions. This is documented in notes/model_comparison.md and notes/ambiguities.md, along with every environment issue encountered and how each was fixed.

### Local development

The code can be written and edited locally (no GPU needed for that), but actual model inference requires a GPU:

```bash
git clone https://github.com/Akansha-Mulchandani/infinia-voice-pipeline.git
cd infinia-voice-pipeline
pip install -r requirements.txt
python pipelines/english/run_chatterbox.py --ref data/reference_audio/mef.wav
```

## Results

Full data in `results/results.csv`. Targets: MOS >= 4.0, speaker similarity >= 0.75, latency < 2000ms (batch), RTF <= 0.5, WER <= 10%.

| Language | Model | MOS | Speaker Similarity | Latency (ms) | RTF | WER | Clones Voice? |
|---|---|---|---|---|---|---|---|
| English | Chatterbox | 4.00 | 0.8897 | 3938.54 | 1.178 | 0.00% | Yes |
| English | CosyVoice2 | 4.62 | 0.9244 | 10335.12* | 2.648* | 0.00% | Yes |
| Arabic | XTTS-v2 | 3.94 | 0.7899 | 17486.24 | 3.279 | 0.00% | Yes |
| Arabic | MMS-TTS | 4.08 | 0.5245** | 147.77 | 0.011 | 22.22% | No |
| Hindi | XTTS-v2 | 4.02 | 0.7970 | 17811.17 | 3.081 | 26.67%*** | Yes |
| Hindi | Indic Parler-TTS | 3.96 | 0.5302** | 9382.00 | 1.628 | 66.67% | No |

\* CosyVoice2 ran CPU-only due to an unresolved `onnxruntime-gpu` install issue. Not directly comparable to the GPU-run numbers above it. See notes/model_comparison.md.

\*\* Low similarity is expected for MMS-TTS and Indic Parler-TTS, neither is a voice-cloning model. Not a bug.

\*\*\* Hindi XTTS-v2 also failed to synthesize 2 of 12 sentences outright, both containing digits. Root cause: the `num2words` library has no Hindi converter. See notes/model_comparison.md.

## Recommended pipeline per language

**English: Chatterbox.** Clones well, fully intelligible, fast on GPU. Misses the latency and RTF targets, but by less than every other model here. (CosyVoice2 scored higher on similarity and MOS but could not be verified on GPU in the time available; worth revisiting if the onnxruntime-gpu issue gets fixed.)

**Arabic: XTTS-v2 for cloning, MMS-TTS when speed matters more than matching a specific voice.** XTTS-v2 is the only Arabic model tested that actually reproduces the reference speaker, at a real cost in speed (RTF 3.28). MMS-TTS is about 90x faster but uses a fixed pre-trained voice.

**Hindi: XTTS-v2**, with the caveat that it cannot currently handle sentences containing digits due to the num2words gap described above. Indic Parler-TTS is not a comparable alternative for cloning, since it does not clone at all.

Full reasoning, every bug found and fixed, and what I would do next: **notes/model_comparison.md**.

## Failure modes (specific, evidenced)

- **Hindi XTTS-v2 fails outright on sentences with digits.** `num2words` has no Hindi converter, so any Hindi sentence containing a number cannot be synthesized. Affected 2 of 12 test sentences (a phone number and a decimal price).
- **CosyVoice2 caps reference audio at 30 seconds** for zero-shot cloning; longer clips are auto-trimmed in this pipeline.
- **MMS-TTS and Indic Parler-TTS cannot clone voices at all**, by design, not as a defect.
- **Round-trip WER measurement is sensitive to punctuation and casing differences** between original text and transcription, which are not real speech errors. Found and fixed for both Arabic (punctuation) and English (casing); documented in case it resurfaces elsewhere.

## What is missing and what I would improve next

- Only 5 of 12 generated sentences per model were used for the latency, RTF, WER, and MOS numbers above, due to time constraints; the remaining 7 per model exist in `results/samples/` but were not run through the metric scripts.
- CosyVoice2 never got proper GPU acceleration (`onnxruntime-gpu`); fixing that is the single most likely thing to change the English recommendation.
- The Hindi digit-handling gap in XTTS-v2 could be worked around by pre-processing Hindi text to spell out numbers before synthesis; not attempted here.
- Did not attempt the brief's bonus items: an Arabic dialect beyond MSA, or a second Indic language beyond Hindi.
- Latency reported is full-clip batch generation time, not true streaming time-to-first-chunk; none of the six models were wired up in streaming mode here.

Full detail on every judgment call made along the way: **notes/ambiguities.md**.

## Closed-source tools used

None for speech generation. All six models (Chatterbox, CosyVoice2, XTTS-v2, MMS-TTS, Indic Parler-TTS) are open-source. Claude (Anthropic) was used as an AI coding assistant throughout for writing and debugging pipeline code, as explicitly permitted by the brief.

## Project structure

```
infinia-voice-pipeline/
├── README.md
├── KAGGLE.md              # exact Kaggle notebook execution steps
├── requirements.txt
├── data/
│   ├── reference_audio/    # my reference voice clip
│   └── eval_sentences/     # 12 test sentences per language
├── pipelines/
│   ├── common/              # shared TTS interface and audio utils
│   ├── english/              # Chatterbox, CosyVoice2
│   ├── arabic/                # XTTS-v2, MMS-TTS
│   └── hindi/                 # XTTS-v2, Indic Parler-TTS
├── eval/                    # 6-metric evaluation harness
├── results/
│   ├── results.csv           # final consolidated results, all 6 models
│   └── samples/               # generated audio clips per model
└── notes/
    ├── model_comparison.md   # full comparison, every bug found and fixed
    └── ambiguities.md         # every assumption and judgment call made
```

## Evaluation metrics

1. **MOS** (1 to 5, human-rated naturalness), target >= 4.0
2. **Speaker similarity** (embedding cosine similarity via Resemblyzer), target >= 0.75
3. **Latency** (median of 5 runs, full-clip batch), target < 2000ms
4. **RTF** (generation time / audio duration), target <= 0.5
5. **WER** (round-trip via faster-whisper large-v3, punctuation/case normalized), target <= 10%
6. **Cross-language consistency check** (flags any missed target per model)

## License

MIT (this repo's code). Individual models retain their own licenses; XTTS-v2 is used here under Coqui's non-commercial CPML (https://coqui.ai/cpml), disclosed explicitly since it is not a fully open commercial license.

## Acknowledgments

Chatterbox by Resemble AI. CosyVoice2 by FunAudioLLM. XTTS-v2 by Coqui AI (maintained fork, coqui-tts). MMS-TTS by Meta. Indic Parler-TTS by AI4Bharat. faster-whisper for round-trip transcription. Resemblyzer for speaker embeddings.
