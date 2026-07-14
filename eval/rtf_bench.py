"""
RTF (Real-Time Factor) benchmark for TTS pipelines.

RTF = generation_time / audio_duration
Target: RTF ≤ 0.5 (generates 2x faster than real-time)
"""

import os
import sys
import argparse

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pipelines', 'common'))

from audio_utils import get_audio_duration


def calculate_rtf(gen_time_sec: float, audio_path: str) -> dict:
    """
    Calculate Real-Time Factor.
    
    Args:
        gen_time_sec: Generation time in seconds
        audio_path: Path to generated audio file
    
    Returns:
        Dict with RTF calculation
    """
    audio_duration = get_audio_duration(audio_path)
    
    if audio_duration == 0:
        return {
            "rtf": None,
            "gen_time_sec": gen_time_sec,
            "audio_duration_sec": 0,
            "error": "Audio duration is zero"
        }
    
    rtf = gen_time_sec / audio_duration
    
    return {
        "rtf": rtf,
        "gen_time_sec": gen_time_sec,
        "audio_duration_sec": audio_duration,
        "error": None
    }


def main():
    parser = argparse.ArgumentParser(description="Calculate RTF for TTS output")
    parser.add_argument("--gen_time", type=float, required=True, help="Generation time in seconds")
    parser.add_argument("--audio", required=True, help="Path to generated audio file")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.audio):
        print(f"Error: Audio file not found: {args.audio}")
        return
    
    results = calculate_rtf(args.gen_time, args.audio)
    
    if results["error"]:
        print(f"Error: {results['error']}")
    else:
        print(f"RTF Results:")
        print(f"  Generation time: {results['gen_time_sec']:.2f}s")
        print(f"  Audio duration:   {results['audio_duration_sec']:.2f}s")
        print(f"  RTF:             {results['rtf']:.4f}")
        
        # Check target
        target_rtf = 0.5
        if results["rtf"] <= target_rtf:
            print(f"  ✓ Meets target (RTF ≤ {target_rtf})")
        else:
            print(f"  ✗ Does not meet target (RTF > {target_rtf})")


if __name__ == "__main__":
    main()
