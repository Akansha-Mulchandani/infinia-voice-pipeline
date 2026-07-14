"""
Speaker similarity benchmark for TTS pipelines.

Computes cosine similarity between speaker embeddings of reference
and generated audio using Resemblyzer.
Target: cosine similarity ≥ 0.75
"""

import os
import sys
import argparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pipelines', 'common'))

from audio_utils import load_audio


def compute_speaker_embedding(audio_path: str, encoder) -> np.ndarray:
    """
    Compute speaker embedding using Resemblyzer.
    
    Args:
        audio_path: Path to audio file
        encoder: Resemblyzer encoder instance
    
    Returns:
        Speaker embedding vector
    """
    # Load audio
    audio, sr = load_audio(audio_path, target_sr=16000)
    
    # Compute embedding
    embedding = encoder.embed_utterance(audio)
    
    return embedding


def compute_similarity(ref_audio_path: str, gen_audio_path: str) -> dict:
    """
    Compute cosine similarity between reference and generated audio.
    
    Args:
        ref_audio_path: Path to reference audio
        gen_audio_path: Path to generated audio
    
    Returns:
        Dict with similarity score
    """
    try:
        from resemblyzer import VoiceEncoder, preprocess_wav
    except ImportError:
        return {
            "cosine_similarity": None,
            "error": "resemblyzer not installed. Install with: pip install resemblyzer"
        }
    
    if not os.path.exists(ref_audio_path):
        return {
            "cosine_similarity": None,
            "error": f"Reference audio not found: {ref_audio_path}"
        }
    
    if not os.path.exists(gen_audio_path):
        return {
            "cosine_similarity": None,
            "error": f"Generated audio not found: {gen_audio_path}"
        }
    
    try:
        # Initialize encoder
        encoder = VoiceEncoder()
        
        # Compute embeddings
        ref_embedding = compute_speaker_embedding(ref_audio_path, encoder)
        gen_embedding = compute_speaker_embedding(gen_audio_path, encoder)
        
        # Reshape for cosine_similarity
        ref_embedding = ref_embedding.reshape(1, -1)
        gen_embedding = gen_embedding.reshape(1, -1)
        
        # Compute cosine similarity
        similarity = cosine_similarity(ref_embedding, gen_embedding)[0][0]
        
        return {
            "cosine_similarity": float(similarity),
            "error": None
        }
        
    except Exception as e:
        return {
            "cosine_similarity": None,
            "error": f"Failed to compute similarity: {str(e)}"
        }


def main():
    parser = argparse.ArgumentParser(description="Compute speaker similarity")
    parser.add_argument("--ref", required=True, help="Reference audio path")
    parser.add_argument("--gen", required=True, help="Generated audio path")
    
    args = parser.parse_args()
    
    results = compute_similarity(args.ref, args.gen)
    
    if results["error"]:
        print(f"Error: {results['error']}")
    else:
        print(f"Speaker Similarity Results:")
        print(f"  Cosine similarity: {results['cosine_similarity']:.4f}")
        
        # Check target
        target_similarity = 0.75
        if results["cosine_similarity"] >= target_similarity:
            print(f"  ✓ Meets target (≥ {target_similarity})")
        else:
            print(f"  ✗ Does not meet target (< {target_similarity})")


if __name__ == "__main__":
    main()
