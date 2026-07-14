"""
Round-trip WER (Word Error Rate) benchmark for TTS pipelines.

Transcribes generated audio back to text and computes WER against original.
Uses faster-whisper (large-v3) for English/Arabic, IndicWhisper for Hindi if needed.
Target: WER ≤ 10%
"""

import os
import sys
import argparse

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pipelines', 'common'))

from audio_utils import load_audio


def transcribe_audio(audio_path: str, language: str = "en") -> dict:
    """
    Transcribe audio using faster-whisper.
    
    Args:
        audio_path: Path to audio file
        language: Language code (en, ar, hi)
    
    Returns:
        Dict with transcription text
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return {
            "text": None,
            "error": "faster-whisper not installed. Install with: pip install faster-whisper"
        }
    
    if not os.path.exists(audio_path):
        return {
            "text": None,
            "error": f"Audio file not found: {audio_path}"
        }
    
    try:
        # Load model (large-v3 for best accuracy)
        # Use GPU if available
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        
        model = WhisperModel("large-v3", device=device, compute_type=compute_type)
        
        # Transcribe
        segments, info = model.transcribe(audio_path, language=language)
        
        # Combine segments
        transcription = " ".join([segment.text for segment in segments])
        
        return {
            "text": transcription.strip(),
            "error": None
        }
        
    except Exception as e:
        return {
            "text": None,
            "error": f"Transcription failed: {str(e)}"
        }


def compute_wer(reference_text: str, hypothesis_text: str) -> dict:
    """
    Compute Word Error Rate using jiwer.
    
    Args:
        reference_text: Original text (ground truth)
        hypothesis_text: Transcribed text
    
    Returns:
        Dict with WER score
    """
    try:
        import jiwer
    except ImportError:
        return {
            "wer": None,
            "error": "jiwer not installed. Install with: pip install jiwer"
        }
    
    try:
        wer = jiwer.wer(reference_text, hypothesis_text)
        
        return {
            "wer": wer,
            "error": None
        }
        
    except Exception as e:
        return {
            "wer": None,
            "error": f"WER computation failed: {str(e)}"
        }


def main():
    parser = argparse.ArgumentParser(description="Compute round-trip WER")
    parser.add_argument("--audio", required=True, help="Generated audio path")
    parser.add_argument("--text", required=True, help="Original text (ground truth)")
    parser.add_argument("--language", default="en", help="Language code (en, ar, hi)")
    
    args = parser.parse_args()
    
    # Transcribe audio
    print(f"Transcribing audio (language: {args.language})...")
    transcribe_result = transcribe_audio(args.audio, args.language)
    
    if transcribe_result["error"]:
        print(f"Transcription error: {transcribe_result['error']}")
        return
    
    hypothesis = transcribe_result["text"]
    print(f"Transcription: {hypothesis}")
    
    # Compute WER
    wer_result = compute_wer(args.text, hypothesis)
    
    if wer_result["error"]:
        print(f"WER error: {wer_result['error']}")
        return
    
    print(f"\nRound-trip WER Results:")
    print(f"  Reference:  {args.text}")
    print(f"  Hypothesis: {hypothesis}")
    print(f"  WER:        {wer_result['wer']:.4f} ({wer_result['wer']*100:.2f}%)")
    
    # Check target
    target_wer = 0.10  # 10%
    if wer_result["wer"] <= target_wer:
        print(f"  ✓ Meets target (≤ {target_wer*100}%)")
    else:
        print(f"  ✗ Does not meet target (> {target_wer*100}%)")


if __name__ == "__main__":
    main()
