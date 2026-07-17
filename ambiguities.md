# Ambiguities and Assumptions

The brief said to make a call, state the assumption, and keep going whenever something was unclear. Here is everything I had to decide on my own, and why.

**Hardware.** Ran everything on Kaggle's free T4 x2 GPU notebooks. I do not have a local GPU (confirmed my laptop only has integrated Intel graphics), so all real inference and benchmarking happened on Kaggle rather than locally. This is disclosed in every results row via the "hardware" column.

**Reference voice.** Used my own voice, recorded on my laptop, roughly 30 to 34 seconds depending on the take. I re-recorded it once partway through the project after the first take produced weak cloning similarity on Chatterbox; the second take (quieter room, more natural pacing, slightly longer) improved similarity meaningfully across every model tested afterward. Content of the reference clip ended up being an unscripted, informal reading rather than the suggested "quick brown fox" script, since I recorded a few different takes and kept whichever sounded cleanest.

**Which Arabic dialect.** Used Modern Standard Arabic only, as allowed by the brief ("Modern Standard Arabic is fine, a dialect is a nice bonus"). Did not attempt a dialect variant given time already spent on environment debugging across six models.

**Which Indic language.** Hindi only, as required ("Hindi at a minimum, more Indic languages is a bonus"). Did not attempt a second Indic language for the same time reason.

**Eval sentence set.** Used 12 sentences per language, covering short conversational phrases, numbers, a name, a phone number, a time, and a decimal price, matching the brief's suggestion. These were auto-generated at project scaffolding time rather than hand-written by me, but I reviewed them for reasonable coverage before running any pipeline against them.

**Subset used for timed metrics.** Latency, RTF, WER, and speaker similarity were computed over the first 5 of the 12 generated samples per model, not all 12, due to time and GPU-hour budget constraints across six separate model environments. MOS ratings were also collected on the same 5-clip subset, from a single rater (myself). The brief allows for a single rater ("you plus a few others" is the target, not a requirement), and I am disclosing that this project used one rater rather than the ideal 2 to 3.

**Running each model in a separate environment.** Several of the models tested here have directly conflicting dependency requirements (most notably, Chatterbox needs a recent `transformers` version that is incompatible with what XTTS-v2's Coqui TTS package expects, and CosyVoice2's own pinned dependencies conflict with almost everything else). Rather than force one shared environment to hold all six models simultaneously, I ran each model in its own Kaggle notebook session and merged the resulting numbers into one final results.csv afterward. This is disclosed explicitly in model_comparison.md rather than presented as if everything ran in one unified pipeline.

**Streaming latency.** The brief's latency target distinguishes streaming (under 500ms to first chunk) from batch (under 2 seconds for a full clip). None of the six models tested here were wired up in a true streaming inference mode; all latency numbers reported are full-clip batch generation time. I judged every model against the batch target only, and did not claim or attempt to measure streaming performance.

**CosyVoice2 running on CPU.** Due to an `onnxruntime-gpu` installation issue that I was not able to resolve within the project's time budget (documented in model_comparison.md), CosyVoice2's ONNX-based components ran on CPU rather than GPU during evaluation. Its latency and RTF numbers are disclosed with this caveat rather than presented as directly comparable to the other, GPU-run models.

**MMS-TTS and Indic Parler-TTS speaker similarity.** Both models scored low on speaker similarity (0.52 and 0.53) because neither is architecturally a voice-cloning model, not because of a bug. I chose to still run the full evaluation harness against both anyway, including the metric that does not really apply to them, rather than skip it, since the brief asked for the same six metrics across every model tested and a low score with a clear explanation is more useful than an omitted one.

**Second candidate models attempted but not deeply pursued.** I made a genuine, sustained attempt at CosyVoice2 (English) across two full sessions and eight distinct dependency blockers before it finally worked. I did not attempt Fish-Speech as a further Arabic candidate, given the amount of time already spent getting the required models working.

**Closed tools.** No closed-source models or APIs were used for any speech generation. Claude (Anthropic) was used as an AI coding assistant throughout, for writing and debugging pipeline code, as explicitly permitted by the brief.
