"""
MOS (Mean Opinion Score) collection for TTS evaluation.

Interactive CLI to collect subjective quality ratings (1-5) from human listeners.
Target: MOS ≥ 4.0 per language.
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path


def collect_mos_ratings(audio_dir: str, output_csv: str = "results/mos_raw.csv") -> None:
    """
    Collect MOS ratings for all audio files in a directory.
    
    Args:
        audio_dir: Directory containing audio files to rate
        output_csv: Path to save ratings CSV
    """
    if not os.path.exists(audio_dir):
        print(f"Error: Audio directory not found: {audio_dir}")
        return
    
    # Get all wav files
    audio_files = sorted(Path(audio_dir).glob("*.wav"))
    
    if not audio_files:
        print(f"No WAV files found in {audio_dir}")
        return
    
    print(f"Found {len(audio_files)} audio files to rate")
    print("Rate each clip on a scale of 1-5:")
    print("  1 = Bad")
    print("  2 = Poor")
    print("  3 = Fair")
    print("  4 = Good")
    print("  5 = Excellent")
    print()
    
    ratings = []
    
    for i, audio_file in enumerate(audio_files):
        print(f"[{i+1}/{len(audio_files)}] {audio_file.name}")
        
        # Play audio (using system default player)
        # Note: This will work differently on different OS
        # On Kaggle, you may need to download clips and listen locally
        print(f"  Play: {audio_file}")
        print(f"  (Listen to the clip, then enter your rating)")
        
        while True:
            try:
                rating = input("  Rating (1-5): ").strip()
                rating = int(rating)
                if 1 <= rating <= 5:
                    break
                else:
                    print("  Please enter a number between 1 and 5")
            except ValueError:
                print("  Please enter a valid number")
        
        ratings.append({
            "audio_file": audio_file.name,
            "audio_path": str(audio_file),
            "rating": rating
        })
        
        print()
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df = pd.DataFrame(ratings)
    df.to_csv(output_csv, index=False)
    
    print(f"Ratings saved to {output_csv}")
    
    # Calculate average
    avg_mos = df["rating"].mean()
    print(f"Average MOS: {avg_mos:.2f}")
    
    # Check target
    target_mos = 4.0
    if avg_mos >= target_mos:
        print(f"✓ Meets target (≥ {target_mos})")
    else:
        print(f"✗ Does not meet target (< {target_mos})")


def compute_average_mos(csv_path: str) -> dict:
    """
    Compute average MOS from a ratings CSV.
    
    Args:
        csv_path: Path to MOS ratings CSV
    
    Returns:
        Dict with average MOS
    """
    if not os.path.exists(csv_path):
        return {
            "average_mos": None,
            "num_ratings": 0,
            "error": f"CSV file not found: {csv_path}"
        }
    
    try:
        df = pd.read_csv(csv_path)
        
        if "rating" not in df.columns:
            return {
                "average_mos": None,
                "num_ratings": 0,
                "error": "CSV must contain 'rating' column"
            }
        
        avg_mos = df["rating"].mean()
        num_ratings = len(df)
        
        return {
            "average_mos": avg_mos,
            "num_ratings": num_ratings,
            "error": None
        }
        
    except Exception as e:
        return {
            "average_mos": None,
            "num_ratings": 0,
            "error": f"Failed to read CSV: {str(e)}"
        }


def main():
    parser = argparse.ArgumentParser(description="Collect MOS ratings")
    parser.add_argument("--audio_dir", required=True, help="Directory containing audio files to rate")
    parser.add_argument("--output", default="results/mos_raw.csv", help="Output CSV path")
    parser.add_argument("--compute", help="Compute average from existing CSV (instead of collecting)")
    
    args = parser.parse_args()
    
    if args.compute:
        # Compute average from existing CSV
        results = compute_average_mos(args.compute)
        
        if results["error"]:
            print(f"Error: {results['error']}")
        else:
            print(f"MOS Results:")
            print(f"  Average MOS: {results['average_mos']:.2f}")
            print(f"  Number of ratings: {results['num_ratings']}")
            
            target_mos = 4.0
            if results["average_mos"] >= target_mos:
                print(f"  ✓ Meets target (≥ {target_mos})")
            else:
                print(f"  ✗ Does not meet target (< {target_mos})")
    else:
        # Collect new ratings
        collect_mos_ratings(args.audio_dir, args.output)


if __name__ == "__main__":
    main()
