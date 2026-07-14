# Ambiguities and Assumptions

This document logs all judgment calls and assumptions made during the development of the infinia-voice-pipeline project, as required by the brief (section 7).

## Language and Dialect Choices

### Arabic Dialect
- **Assumption**: Modern Standard Arabic (MSA) is used for all Arabic evaluation sentences
- **Rationale**: MSA is the most widely understood written form of Arabic across the Arab world
- **Alternative considered**: Egyptian Arabic, Gulf Arabic - rejected due to narrower regional understanding
- **Impact**: Model performance may vary significantly for regional dialects

### Hindi Script
- **Assumption**: Hindi is written in Devanagari script in evaluation sentences
- **Rationale**: Standard formal Hindi uses Devanagari script
- **Impact**: Models may handle transliterated Hindi differently

## Hardware Assumptions

### Target Hardware
- **Assumption**: Kaggle free tier GPU (T4 x2, 16GB VRAM) as primary target
- **Rationale**: User specified this as the execution environment
- **Impact**: Models that require more VRAM will need to be downgraded or excluded

### Local Development
- **Assumption**: Local development is CPU-only (Windows 11 with Intel Iris Xe integrated graphics)
- **Rationale**: User specified no local GPU
- **Impact**: All model inference and benchmarking must happen on Kaggle, not locally

## Model Selection Assumptions

### Chatterbox (English)
- **Assumption**: chatterbox-tts package provides zero-shot voice cloning API
- **Uncertainty**: The actual API may differ from the implementation
- **Fallback**: If Chatterbox fails, CosyVoice2 is the alternative English model

### CosyVoice2 (English)
- **Assumption**: CosyVoice2 supports zero-shot voice cloning for English
- **Uncertainty**: Installation via git from FunAudioLLM/CosyVoice may fail if repo structure changes
- **Fallback**: Chatterbox is the primary English model

### XTTS-v2 (Arabic/Hindi)
- **Assumption**: XTTS-v2 supports Arabic and Hindi in the installed version
- **Uncertainty**: Language support varies by version of Coqui TTS
- **Language codes**: Used "ar" for Arabic and "hi" for Hindi - these may need verification
- **Fallback**: If XTTS-v2 doesn't support a language, the alternative model (MMS-TTS for Arabic, Indic Parler-TTS for Hindi) becomes primary

### MMS-TTS (Arabic)
- **Assumption**: MMS-TTS does NOT support zero-shot voice cloning (documented limitation)
- **Rationale**: MMS-TTS is designed as a fixed single-speaker model per language
- **Impact**: Speaker similarity metrics will be meaningless for this model
- **Decision**: Included anyway to benchmark synthesis quality (not voice cloning)

### Indic Parler-TTS (Hindi)
- **Assumption**: Indic Parler-TTS supports voice cloning
- **Uncertainty**: Some Indic TTS models are single-speaker only
- **Fallback**: XTTS-v2 is the alternative Hindi model

## Evaluation Metric Assumptions

### MOS Collection
- **Assumption**: MOS requires 2-3 human listeners per language for reliable results
- **Current state**: Placeholder implementation - no actual ratings collected yet
- **Impact**: MOS column in results will be empty until human ratings are collected
- **Alternative**: Could use automated quality metrics (e.g., NISQA) as proxy

### Speaker Similarity
- **Assumption**: Resemblyzer speaker embeddings are appropriate for all languages
- **Uncertainty**: Resemblyzer may be biased toward English speakers
- **Alternative**: SpeechBrain ECAPA-TDNN could be used instead
- **Target**: Set cosine similarity target to ≥0.75 based on typical voice cloning benchmarks

### Latency Measurement
- **Assumption**: Measured as full clip generation time (batch), not streaming latency
- **Rationale**: Streaming API may not be available for all models
- **Target**: Set batch target to <2s, streaming target to <500ms
- **Future improvement**: Could implement actual streaming latency measurement

### WER Calculation
- **Assumption**: faster-whisper large-v3 works adequately for Arabic and Hindi
- **Uncertainty**: Whisper may have lower accuracy on Indic accents
- **Alternative**: IndicWhisper could be used for Hindi if accuracy is poor
- **Target**: Set WER target to ≤10% based on typical TTS benchmarks

### RTF Target
- **Assumption**: RTF ≤0.5 (2x faster than real-time) is achievable on T4 GPU
- **Rationale**: Common target for real-time TTS systems
- **Impact**: Models with RTF >0.5 may not be suitable for real-time applications

## Sentence Set Choices

### English Sentences
- **Count**: 12 sentences
- **Content**: Includes numbers (single, double, triple digits), phone numbers, proper names (Alex, Sarah), time expressions
- **Rationale**: Covers common use cases and edge cases for TTS
- **Alternative considered**: More complex sentences, longer paragraphs - rejected to keep evaluation time reasonable

### Arabic Sentences
- **Count**: 12 sentences
- **Content**: Similar structure to English (numbers, names, time expressions)
- **Translation**: Direct translations of English sentences where possible
- **Impact**: May not reflect natural Arabic phrasing in all cases

### Hindi Sentences
- **Count**: 12 sentences
- **Content**: Similar structure to English (numbers, names, time expressions)
- **Translation**: Direct translations of English sentences where possible
- **Impact**: May not reflect natural Hindi phrasing in all cases

## Implementation Assumptions

### Pipeline Interface
- **Assumption**: All pipelines can implement the same TTSPipeline interface
- **Rationale**: Enables consistent evaluation across models
- **Impact**: Models with fundamentally different APIs may need wrapper classes

### Audio Format
- **Assumption**: 16kHz mono WAV is the standard format for all models
- **Rationale**: Common format supported by most TTS models
- **Impact**: Models requiring different sample rates will need resampling

### Batch Processing
- **Assumption**: Batching all eval sentences per model is more efficient than individual runs
- **Rationale**: Reduces model loading overhead
- **Impact**: May hide per-sentence variability in performance

## Kaggle Execution Assumptions

### Session Persistence
- **Assumption**: Kaggle session state does NOT persist between sessions
- **Rationale**: Kaggle's ephemeral notebook environment
- **Impact**: Model checkpoints must be saved as Kaggle Datasets for reuse

### Internet Access
- **Assumption**: Internet access is required for model downloads
- **Rationale**: HuggingFace and GitHub downloads need internet
- **Impact**: If Kaggle disables internet, models must be pre-downloaded as Datasets

### GPU Utilization
- **Assumption**: T4 x2 GPU is sufficient for all models
- **Rationale**: User specified this hardware
- **Impact**: Models requiring more VRAM may need CPU fallback or smaller checkpoints

## Missing Information

### Model Documentation
- **Gap**: Official documentation for Chatterbox, CosyVoice2, and Indic Parler-TTS APIs may be incomplete or outdated
- **Impact**: Implementations are based on best guesses and may need adjustment
- **Mitigation**: Added clear error messages and logging to identify API mismatches

### Language Support Verification
- **Gap**: XTTS-v2 language support for Arabic and Hindi not verified
- **Impact**: May need to adjust language codes or switch to alternative models
- **Mitigation**: Added language support checks in pipeline load methods

### Voice Cloning Verification
- **Gap**: Voice cloning support not verified for Indic Parler-TTS
- **Impact**: May discover it's single-speaker only during evaluation
- **Mitigation**: Documented this uncertainty in README and pipeline comments

## Future Decisions Needed

### Bonus Indic Language
- **Decision needed**: Which Indic language to add as bonus (Tamil/Bengali/Marathi)?
- **Factors**: Model availability, voice cloning support, evaluation time
- **Timeline**: Not implemented in current version

### MOS Collection Method
- **Decision needed**: How to collect MOS ratings efficiently?
- **Options**: Web interface, local script, crowdsourcing platform
- **Current state**: Placeholder CLI script - needs actual human raters

### Model Router
- **Decision needed**: Should we implement automatic model selection per language?
- **Factors**: Performance, latency, quality trade-offs
- **Current state**: Manual selection - could be automated in future

## Conclusion

This project required making numerous assumptions due to:
1. Fast-moving open-source TTS landscape (APIs change frequently)
2. Limited documentation for some models
3. Language support uncertainty in multilingual models
4. Hardware constraints (Kaggle free tier)

All assumptions are documented here to enable reproducibility and future refinement. When actual benchmark results are available from Kaggle, these assumptions should be revisited and updated based on real performance data.
