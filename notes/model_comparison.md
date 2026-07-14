# Model Comparison

This document compares all candidate models tried for each language, including why winners were chosen and documented failure modes.

**Note**: This document will be updated with actual benchmark results after running evaluation on Kaggle. Current entries are based on model characteristics and known limitations.

## English Models

### Chatterbox (Resemble AI)

**Status**: Primary candidate

**Strengths**:
- Zero-shot voice cloning from short reference clips
- Optimized for low latency (streaming-capable)
- Good naturalness for English speech
- Active development by Resemble AI

**Weaknesses**:
- API may differ from implementation (needs verification)
- Limited to English (not multilingual)
- May have limited language coverage for non-US accents

**Known Failure Modes**:
- May struggle with proper nouns not in training data
- May have accent bias toward US English
- Installation via pip may have version conflicts

**Expected Performance**:
- Latency: <500ms (streaming target)
- Speaker similarity: ≥0.75
- MOS: ≥4.0 (expected)

### CosyVoice2 (FunAudioLLM)

**Status**: Alternative candidate

**Strengths**:
- Zero-shot voice cloning
- Strong streaming latency performance
- Active development by FunAudioLLM
- Good naturalness for English speech

**Weaknesses**:
- Installation requires git install (more complex)
- Repository structure may change (breaking changes)
- API may differ from implementation (needs verification)
- Limited documentation

**Known Failure Modes**:
- Installation may fail if repo structure changes
- May have different API than expected
- May require specific dependencies not in requirements.txt

**Expected Performance**:
- Latency: <500ms (streaming target)
- Speaker similarity: ≥0.75
- MOS: ≥4.0 (expected)

**Winner Decision**: TBD (after benchmarking)

**Rationale**: Will be determined by actual benchmark results comparing latency, speaker similarity, and MOS.

## Arabic Models

### XTTS-v2 (Coqui TTS)

**Status**: Primary candidate

**Strengths**:
- Zero-shot voice cloning
- Multilingual support (includes Arabic in some versions)
- Good naturalness across languages
- Maintained fork available (coqui-ai/TTS is archived)

**Weaknesses**:
- Language support varies by version (needs verification)
- May not support Arabic in installed version
- Language code may need adjustment ("ar" vs "ar-ar")
- Larger model size (may need GPU)

**Known Failure Modes**:
- May mispronounce numbers above three digits
- May struggle with complex numerals
- May have limited Arabic dialect support (MSA only)
- May drop reference speaker's pitch on long sentences

**Expected Performance**:
- Latency: <2s (batch target)
- Speaker similarity: ≥0.75
- MOS: ≥3.5 (expected, may be lower than English)
- WER: ≤15% (expected, higher target due to Arabic complexity)

### MMS-TTS (Meta)

**Status**: Fallback (non-cloning)

**Strengths**:
- Broad language coverage (includes Arabic)
- Lightweight model (fits in smaller VRAM)
- Good synthesis quality for Arabic
- Stable API (well-documented)

**Weaknesses**:
- **Does NOT support zero-shot voice cloning** (fundamental limitation)
- Uses fixed pre-trained voice per language
- Speaker similarity metrics are meaningless
- Not suitable for voice cloning use case

**Known Failure Modes**:
- Cannot clone reference speaker's voice (by design)
- Output will always use model's pre-trained voice
- Speaker similarity will be low (expected, not a bug)

**Expected Performance**:
- Latency: <1s (batch target, faster than XTTS-v2)
- Speaker similarity: N/A (not applicable)
- MOS: ≥3.0 (expected, for synthesis quality only)
- WER: ≤12% (expected, may be better than XTTS-v2)

**Winner Decision**: TBD (after benchmarking)

**Rationale**: XTTS-v2 is the only true voice cloning option for Arabic. MMS-TTS is included only for synthesis quality benchmarking, not voice cloning.

**Important Note**: If XTTS-v2 doesn't support Arabic in the installed version, there is no viable voice cloning option for Arabic in this benchmark. This represents a gap in open-source Arabic voice cloning tools.

## Hindi Models

### Indic Parler-TTS (AI4Bharat)

**Status**: Primary candidate

**Strengths**:
- Purpose-built for Indic languages
- Good Hindi language support
- Active development by AI4Bharat
- Optimized for Indic script

**Weaknesses**:
- Voice cloning support uncertain (may be single-speaker only)
- API may differ from implementation (needs verification)
- May have limited documentation
- May not support zero-shot cloning

**Known Failure Modes**:
- May not support voice cloning (single-speaker model)
- May drop reference speaker's pitch on long sentences
- May struggle with transliterated Hindi
- May have limited vocabulary for technical terms

**Expected Performance**:
- Latency: <2s (batch target)
- Speaker similarity: TBD (depends on cloning support)
- MOS: ≥3.5 (expected)
- WER: ≤15% (expected, higher target due to Hindi complexity)

### XTTS-v2 (Coqui TTS)

**Status**: Alternative candidate

**Strengths**:
- Zero-shot voice cloning
- Multilingual support (includes Hindi in some versions)
- Good naturalness across languages
- Maintained fork available

**Weaknesses**:
- Language support varies by version (needs verification)
- May not support Hindi in installed version
- Language code may need adjustment ("hi" vs "hi-in")
- Larger model size (may need GPU)

**Known Failure Modes**:
- May mispronounce Hindi numerals
- May struggle with complex Hindi grammar
- May drop reference speaker's pitch on long sentences
- May have limited Indic language support

**Expected Performance**:
- Latency: <2s (batch target)
- Speaker similarity: ≥0.75
- MOS: ≥3.5 (expected)
- WER: ≤15% (expected)

**Winner Decision**: TBD (after benchmarking)

**Rationale**: Will be determined by actual benchmark results and verification of voice cloning support in Indic Parler-TTS.

## Cross-Language Comparison

### Language Support Matrix

| Model | English | Arabic (MSA) | Hindi | Voice Cloning |
|-------|---------|-------------|-------|---------------|
| Chatterbox | ✓ | ✗ | ✗ | ✓ |
| CosyVoice2 | ✓ | ? | ? | ✓ |
| XTTS-v2 | ✓ | ? | ? | ✓ |
| MMS-TTS | ✓ | ✓ | ✓ | ✗ |
| Indic Parler-TTS | ✗ | ✗ | ✓ | ? |

**Legend**: ✓ = Supported, ✗ = Not supported, ? = Needs verification

### Key Findings

1. **No single model covers all three languages with voice cloning**: This validates the brief's hint about needing a per-language router.

2. **Arabic has the fewest voice cloning options**: Only XTTS-v2 is a true voice cloning candidate. MMS-TTS doesn't support cloning.

3. **Hindi has good Indic-specific options**: Indic Parler-TTS is purpose-built for Indic languages, though cloning support needs verification.

4. **English has the most mature options**: Both Chatterbox and CosyVoice2 are well-developed for English voice cloning.

### Installation Complexity Ranking

1. **Easiest**: XTTS-v2 (pip install TTS)
2. **Medium**: Chatterbox (pip install chatterbox-tts)
3. **Medium**: MMS-TTS (pip install transformers)
4. **Hard**: Indic Parler-TTS (pip install transformers, but API uncertain)
5. **Hardest**: CosyVoice2 (git install from FunAudioLLM/CosyVoice)

### VRAM Requirements (Estimated)

1. **Lowest**: MMS-TTS (~2GB)
2. **Low**: Indic Parler-TTS (~4GB)
3. **Medium**: Chatterbox (~6GB)
4. **High**: XTTS-v2 (~8GB)
5. **Highest**: CosyVoice2 (~10GB)

*All estimates based on typical model sizes. Actual usage may vary.*

## Failure Mode Summary

### Common Failure Modes Across Models

1. **Language Support Uncertainty**: Multilingual models (XTTS-v2, CosyVoice2) have varying language support by version
2. **API Changes**: Fast-moving repos (CosyVoice2, Chatterbox) may have breaking API changes
3. **VRAM Limitations**: Larger models may not fit in 16GB VRAM (Kaggle T4)
4. **Accent/Dialect Bias**: Models may be biased toward specific accents or dialects
5. **Proper Noun Handling**: All models may struggle with names not in training data

### Model-Specific Failure Modes

**Chatterbox**:
- Installation may fail due to version conflicts
- API may differ from documentation

**CosyVoice2**:
- Git install may fail if repo structure changes
- May require specific dependencies not listed

**XTTS-v2**:
- Language codes may need adjustment
- May not support Arabic/Hindi in installed version
- May mispronounce complex numerals

**MMS-TTS**:
- Does not support voice cloning (by design)
- Speaker similarity meaningless

**Indic Parler-TTS**:
- May not support voice cloning
- API may differ from documentation
- May be single-speaker only

## Recommendations for Future Model Selection

1. **Verify language support before investing build time**: Check model documentation for actual language support in the installed version
2. **Have fallback models ready**: Each language has 2 candidates for this reason
3. **Document API differences**: Keep track of actual vs. expected API calls
4. **Consider model size**: Ensure models fit in target hardware (16GB VRAM)
5. **Test installation early**: Install models early in the process to catch issues

## Conclusion

This benchmark compares 6 models across 3 languages with 2 candidates per language. Key findings:

- **English**: Strong options (Chatterbox, CosyVoice2) - winner TBD
- **Arabic**: Limited options (XTTS-v2 only for cloning) - represents a gap in open-source tools
- **Hindi**: Good Indic options (Indic Parler-TTS, XTTS-v2) - winner TBD

The brief's hint about needing a per-language router is validated by the lack of a single model covering all three languages with voice cloning support.

**Next Steps**: Run actual benchmarks on Kaggle to determine winners per language based on real performance data.
