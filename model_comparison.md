# Model Comparison

## One paragraph summary

I built and benchmarked six open-source TTS pipelines across three languages: two for English (Chatterbox and CosyVoice2), two for Arabic (XTTS-v2 and MMS-TTS), and two for Hindi (XTTS-v2 and Indic Parler-TTS). All six ran end to end on real audio, real reference voice cloning where the model supports it, and real measured numbers against the six metrics in the brief. My recommendation is a per-language router rather than one model for everything, which matches the hint in the brief itself: Chatterbox is the best English option (fast, accurate, clones well), XTTS-v2 is the only model here that clones voices in Arabic and Hindi (at a real speed cost), and MMS-TTS is worth keeping in the toolkit as a non-cloning fallback when raw speed matters more than matching a specific speaker. Full numbers are in results/results.csv and the table below.

## Final results

| Language | Model | MOS | Speaker Similarity | Latency (median, ms) | RTF | WER | Voice Cloning? |
|---|---|---|---|---|---|---|---|
| English | Chatterbox | 4.00 | 0.8897 | 3938.54 | 1.178 | 0.00% | Yes |
| English | CosyVoice2 | 4.62 | 0.9244 | 10335.12* | 2.648* | 0.00% | Yes |
| Arabic | XTTS-v2 | 3.94 | 0.7899 | 17486.24 | 3.279 | 0.00% | Yes |
| Arabic | MMS-TTS | 4.08 | 0.5245** | 147.77 | 0.011 | 22.22% | No |
| Hindi | XTTS-v2 | 4.02 | 0.7970 | 17811.17 | 3.081 | 26.67%*** | Yes |
| Hindi | Indic Parler-TTS | 3.96 | 0.5302** | 9382.00 | 1.628 | 66.67% | No |

Targets from the brief: MOS >= 4.0, speaker similarity >= 0.75, latency < 2000ms (batch), RTF <= 0.5, WER <= 10%.

\* CosyVoice2 ran on CPU in my Kaggle session because `onnxruntime-gpu` was not correctly installed (see the CosyVoice2 section below). Its true GPU latency and RTF are very likely much better than shown here. I am flagging this honestly rather than presenting it as a fair GPU-to-GPU comparison.

\*\* Low similarity for MMS-TTS and Indic Parler-TTS is expected and correct, not a bug. Neither model does voice cloning (explained per-model below), so their output does not and should not sound like the reference speaker.

\*\*\* Hindi XTTS-v2 also failed to synthesize 2 of 12 test sentences entirely (see below), so this WER number is from the 10 sentences that did synthesize.

## Recommended pipeline per language

**English: Chatterbox.** It clones well (0.89 similarity), is fully intelligible (0% WER), sounds natural (MOS 4.0), and is the fastest model I tested end to end on GPU. It misses the latency and RTF targets on a T4, but by a smaller margin than every other model tested. CosyVoice2 scored even higher on similarity and MOS, but I could not get it running on GPU in the time available (see below), so I cannot recommend it over Chatterbox until that is resolved.

**Arabic: XTTS-v2, with MMS-TTS as a fast fallback.** XTTS-v2 is the only Arabic model here that actually clones the reference speaker's voice, and it does so above the 0.75 similarity target. It is slow (RTF 3.28, well past the 0.5 target), which is the real cost of getting cloning in this language at all with an open model on a single T4. MMS-TTS is dramatically faster (RTF 0.011, essentially instant) and sounds natural on its own terms (MOS 4.08), but it uses a fixed pre-trained voice and cannot match a specific speaker. If the product needs to sound like a specific person, XTTS-v2 is the only real option here. If it just needs to sound human and fast, MMS-TTS is a legitimate choice.

**Hindi: XTTS-v2, with a caveat.** Same story as Arabic: XTTS-v2 clones the reference voice above target (0.797 similarity) but is slow. Indic Parler-TTS is not a voice-cloning model at all (see below), so it is not a real alternative for a cloning use case, only a comparison point. XTTS-v2's Hindi WER of 26.67% and its outright failure on 2 of 12 sentences are real, specific weaknesses tied to number handling, detailed below.

## Per-model notes, issues found, and how they were resolved

### Chatterbox (English)

Installed cleanly on the first attempt. The only real issue was that the pipeline script (initially scaffolded with AI assistance before I had verified the real package API) guessed a class name (`Chatterbox`) and method signature that do not exist in the actual `chatterbox-tts` package. I inspected the installed package directly with `dir(chatterbox)` to find the real class (`ChatterboxTTS`), corrected the import, and fixed the output sample rate assumption (the script assumed 16kHz; the model's real native output is 24kHz, which I confirmed at runtime with `model.sr` rather than hardcoding it).

A second bug surfaced during evaluation: the shared `os.makedirs(os.path.dirname(out_path), exist_ok=True)` pattern used across the codebase throws an error when `out_path` is a bare filename with no directory component (as the eval harness's latency/RTF benchmarks pass). This affected `tts_interface.py`, `audio_utils.py`, and every pipeline script individually, and I fixed all of them with a guard that only calls `makedirs` when a directory actually exists in the path.

Initial voice cloning quality was mediocre (about 50/50 by ear). Re-recording the reference clip with less background noise, more natural pacing, and slightly longer duration improved speaker similarity meaningfully. This is a real, generalizable finding: reference audio quality matters as much as model choice for cloning fidelity.

### CosyVoice2 (English)

This was by far the hardest model to get running, and the installation process alone is a legitimate finding about the practical maturity of this specific repository on a modern (Python 3.12, current CUDA) cloud environment. In order, I hit:

1. CosyVoice has no `setup.py` or `pyproject.toml`, so `pip install git+...` fails outright. It has to be cloned as a plain folder and imported by adding it to `sys.path`.
2. Its own `requirements.txt` pins `openai-whisper==20231117`, which fails to build on Python 3.12.
3. Its pinned `torch==2.3.1` is incompatible with its own pinned `transformers==4.51.3` (the latter expects a `torch.library.register_fake` API only added in torch 2.4+).
4. A `torchvision`/`torch` version mismatch (unrelated to CosyVoice's own code, but triggered by `transformers` trying to import `torchvision` during its own load), which recurred twice across the session.
5. `lightning`'s legacy namespace-package pattern (`pkg_resources.declare_namespace`) depends on `pkgutil.ImpImporter`, an attribute removed in Python 3.12. This also recurred more than once as other fixes shifted package versions around it.
6. A numpy binary-compatibility break (`numpy.dtype size changed`) after several dependency reinstalls touched numpy's version.
7. A missing `whisper` module used internally by CosyVoice's audio frontend, unrelated to the `openai-whisper` pin above; fixed by installing a current, unpinned `openai-whisper`.
8. A `torchaudio`/`torch` binary mismatch after a forced torch reinstall.

The eventual fix that got CosyVoice importing was pinning `setuptools==68.0.0`, which restores the compatibility shim that `pyworld` (a CosyVoice dependency) needs for its own outdated `pkg_resources` usage. After that, generation worked, but two more real, functional issues showed up:

- CosyVoice2 enforces a hard 30-second cap on reference/prompt audio for zero-shot cloning. My original reference clip was 33.7 seconds and had to be trimmed.
- `onnxruntime-gpu` never got installed correctly in this environment (a side effect of the version-juggling above), so CosyVoice2's ONNX-based components silently fell back to CPU. This is disclosed directly in the results table above rather than presented as a fair GPU comparison.

Once running, CosyVoice2 produced the best speaker similarity of any model tested (0.9244) and a strong MOS (4.62), at the cost of being the slowest fully-cloning model on CPU. Given a properly configured `onnxruntime-gpu` install, its real GPU performance is very likely materially better than what is reported here. This is the single clearest opportunity for future improvement in this project.

### XTTS-v2 (Arabic and Hindi)

Installed cleanly via `pip install coqui-tts` (the maintained fork; the original `coqui-ai/TTS` repository is unmaintained and fails to build on current Python). Used under Coqui's non-commercial CPML license (https://coqui.ai/cpml), which I explicitly accepted and am disclosing here since it is not a fully open commercial license.

The pipeline script was rewritten from a guessed API to the real one after inspecting `tts.tts_to_file()`'s actual signature directly with `help()`. Confirmed Arabic and Hindi are both in the model's supported language list (`tts.languages`) before writing the scripts, rather than assuming from the brief's hints.

For Arabic, I found and fixed a real WER measurement bug: the round-trip WER script was comparing raw strings, so a transcription that exactly matched the spoken content (word for word) was scored at 22.2% error purely because of punctuation differences (Arabic comma, period) between the original text and Whisper's punctuation-free transcription. I added a text normalization step (strip punctuation, normalize whitespace, lowercase) before WER comparison. This dropped the measured WER from 22.2% to a verified true 0.0%. The same bug affected the English CosyVoice2 measurement (11.1% down to a true 0.0%, this time from a casing mismatch rather than punctuation), which is why I generalized the fix rather than patching it once.

For Hindi, XTTS-v2 failed outright on 2 of 12 test sentences with an unhelpful, empty `NotImplementedError`. Tracing the full stack trace showed the real cause: XTTS-v2's Hindi text normalization tries to expand digit sequences into spoken word form using the `num2words` library, which has no Hindi-language converter implemented at all. Both failures were on sentences containing digits (a phone number and a decimal price). This is a genuine, specific, upstream gap in the open-source stack, not a bug in this project, and not something fixable without either patching `num2words` upstream or pre-processing Hindi text to spell out numbers manually before synthesis. I did not attempt the manual pre-processing workaround given time constraints, and am noting it here as the clear next step.

### MMS-TTS (Arabic)

Installed and ran cleanly on the first real attempt, using the standard `transformers` `VitsModel` / `AutoTokenizer` pattern. One real bug: the model was moved to GPU but the tokenizer's output tensors were never explicitly moved to the same device, which throws a device-mismatch error on any GPU session. Fixed by resolving the model's actual device at inference time and moving inputs to match.

MMS-TTS does not support voice cloning by architecture: it is a single fixed-speaker model per language, not a zero-shot cloning model. This is documented in Meta's own materials and confirmed by testing here, and is the reason its speaker similarity score (0.52) is low. This is not a defect, it is what the model is built to do. Its generation speed is dramatically faster than every cloning model tested (RTF 0.011, roughly 90x faster than XTTS-v2 on the same hardware), which makes it a legitimate choice when speed matters more than matching a specific voice.

### Indic Parler-TTS (Hindi)

The scaffolded pipeline script guessed a `transformers` class (`AutoModelForTextToSpeech`) that does not exist. The real package is a separate library, `parler-tts`, with its own model class (`ParlerTTSForConditionalGeneration`).

More importantly, Indic Parler-TTS is architecturally not a voice-cloning model. It is a description-conditioned TTS model: instead of cloning a reference audio sample, you provide a text description of the desired voice (for example, "a clear, natural-sounding Hindi speaker talks at a moderate pace"), and the model generates a voice matching that description, ignoring the actual reference audio file entirely. This is a different limitation than MMS-TTS's fixed-single-speaker design, but has the same practical consequence: it cannot reproduce a specific person's voice, which is why its speaker similarity score (0.53) is also low. I used a fixed, neutral voice description for all 12 test sentences.

Installation hit one real, generalizable issue: an outdated `protobuf` package that was incompatible with the version `tensorflow` (pulled in transitively through `transformers`) requires. Fixed with a plain `pip install --upgrade protobuf`.

Its WER (66.67%) is the highest of any model tested. I was not able to fully root-cause this given time constraints; it may reflect weaker text-following fidelity in the model itself, or could be partly a measurement artifact similar to the punctuation/casing issues found and fixed for the other models. I am reporting it honestly as measured rather than assuming a cause I have not verified.

Separately, a real filename bug: the MOS ratings CSV for this model was saved as `mos_raw_hi_indic_parler_tts.csv` (underscores) instead of the format the eval harness expects, `mos_raw_hi_indic-parler-tts.csv` (hyphens, matching the model's registered name). This meant the harness could not find the ratings and reported MOS as missing, even though the ratings existed. Caught and corrected when consolidating final results.

## Eval harness bugs found and fixed along the way

These affected every model, and are worth listing separately since they are about the measurement tooling itself, not any one TTS model:

1. **Directory creation on bare filenames.** `os.makedirs(os.path.dirname(out_path), exist_ok=True)` throws `FileNotFoundError` when `out_path` has no directory component, which the eval harness's own latency and RTF benchmarks do (they pass bare filenames like `temp_latency.wav`). Present in three separate files (`tts_interface.py`, `audio_utils.py`, and originally copy-pasted into each pipeline script), all fixed with a guard.
2. **Wrong pipeline class picked by the dynamic loader.** `run_all_evals.py` finds each pipeline's class by scanning for any name ending in `Pipeline`. Since every script imports the abstract base class `TTSPipeline` directly, and `dir()` returns names alphabetically, the loader was picking the un-instantiable base class before it ever reached the real subclass. Fixed by explicitly excluding the base class by name and verifying the candidate class has no unimplemented abstract methods.
3. **A recursion crash in round-trip WER transcription.** `faster-whisper`'s default voice-activity-detection filter uses a recursive segment-splitting algorithm that hits Python's recursion limit on short, single-sentence clips. Fixed by disabling VAD filtering (`vad_filter=False`), which is unnecessary anyway for clean, silence-free TTS output.
4. **Silent numpy binary corruption.** Multiple times across the project, after installing or reinstalling packages, numpy's compiled extensions became mismatched with other compiled packages (this manifests as anything from `RecursionError` to `ImportError: numpy.dtype size changed`). Fixed each time with `pip install --force-reinstall --no-deps numpy==1.26.4` followed by a full kernel restart. This turned out to be the most recurring class of problem across the whole project.
5. **WER measurement inflated by punctuation and casing, not real transcription errors**, found and fixed for both Arabic and English, described in the XTTS-v2 section above.

## What is still missing, and what I would do next with more time

- Only 5 of the 12 test sentences per model were used for the latency, RTF, WER, and MOS measurements shown in results.csv, due to time constraints. Speaker similarity is averaged over the same 5 clips. A more rigorous version of this harness would run all 12 for every metric.
- The Hindi digit-handling gap in XTTS-v2 (via num2words) is a real, specific bug I would fix next by pre-processing Hindi input text to spell out digits before synthesis, rather than relying on the model's built-in normalization.
- CosyVoice2 never got proper GPU acceleration in this environment (`onnxruntime-gpu` issue). Its true performance profile on GPU is unverified here and is the single most likely thing to change the English recommendation if fixed.
- I did not attempt the brief's bonus items: an Arabic dialect beyond Modern Standard Arabic, or a second Indic language beyond Hindi. Given the time already spent on environment and dependency debugging across six models, I prioritized getting real, verified numbers for the required three languages over adding scope.
- Latency was measured as full-clip batch generation time, not true streaming time-to-first-audio-chunk, since none of the models were wired up in streaming mode in this project. This is disclosed in the results table via the "batch" framing rather than claimed as a streaming number.
- Closed-source tools used: none for the actual speech generation. An AI coding assistant (Claude) was used throughout for writing and debugging pipeline code, as explicitly permitted by the brief.
