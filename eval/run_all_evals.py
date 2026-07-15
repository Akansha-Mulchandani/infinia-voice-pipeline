"""
Evaluation orchestrator for TTS pipelines.

Runs all 6 evaluation metrics (MOS, speaker similarity, latency, RTF, WER)
and aggregates results into results/results.csv.
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pipelines', 'common'))

from audio_utils import get_audio_duration

# Import benchmark modules
from latency_bench import measure_latency
from rtf_bench import calculate_rtf
from speaker_similarity import compute_similarity
from round_trip_wer import transcribe_audio, compute_wer
from mos_collect import compute_average_mos


def evaluate_pipeline(
    language: str,
    model: str,
    pipeline_module: str,
    reference_audio: str,
    samples_dir: str,
    eval_sentences_file: str,
    hardware: str = "T4"
) -> dict:
    """
    Run full evaluation on a single pipeline.
    
    Args:
        language: Language code (en, ar, hi)
        model: Model name (e.g., chatterbox, xtts-v2)
        pipeline_module: Path to pipeline module
        reference_audio: Path to reference audio
        samples_dir: Directory containing generated samples
        eval_sentences_file: Path to eval sentences file
        hardware: Hardware identifier
    
    Returns:
        Dict with all evaluation results
    """
    print(f"\n{'='*60}")
    print(f"Evaluating: {language} - {model}")
    print(f"{'='*60}")
    
    results = {
        "language": language,
        "model": model,
        "hardware": hardware,
        "MOS": None,
        "speaker_cosine_sim": None,
        "same_speaker_human_judgment": None,  # To be filled manually
        "latency_ms": None,
        "RTF": None,
        "WER": None,
        "notes": ""
    }
    
    # Load eval sentences
    if not os.path.exists(eval_sentences_file):
        results["notes"] += f"Eval sentences file not found: {eval_sentences_file}. "
        print(f"Warning: {results['notes']}")
        return results
    
    with open(eval_sentences_file, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    if not sentences:
        results["notes"] += "No eval sentences found. "
        print(f"Warning: {results['notes']}")
        return results
    
    # Check if samples exist
    if not os.path.exists(samples_dir):
        results["notes"] += f"Samples directory not found: {samples_dir}. "
        print(f"Warning: {results['notes']}")
        return results
    
    sample_files = sorted(Path(samples_dir).glob("*.wav"))
    if not sample_files:
        results["notes"] += "No sample files found. "
        print(f"Warning: {results['notes']}")
        return results
    
    # Use first sample for latency, RTF, WER benchmarks
    first_sample = sample_files[0]
    first_sentence = sentences[0] if sentences else ""
    
    # Import pipeline
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("pipeline_module", pipeline_module)
        pipeline_module_obj = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pipeline_module_obj)
        
        # Get pipeline class
        pipeline_class = None
        for name in dir(pipeline_module_obj):
            if name.endswith('Pipeline') and name != 'TTSPipeline':
                candidate = getattr(pipeline_module_obj, name)
                # Skip the abstract base class itself and anything that
                # isn't actually a usable subclass
                if isinstance(candidate, type) and not getattr(candidate, '__abstractmethods__', None):
                    pipeline_class = candidate
                    break
        
        if not pipeline_class:
            results["notes"] += "Could not find pipeline class. "
            print(f"Warning: {results['notes']}")
            return results
        
        # Initialize pipeline
        pipeline = pipeline_class()
        pipeline.load(reference_audio)
        
    except Exception as e:
        results["notes"] += f"Failed to load pipeline: {str(e)}. "
        print(f"Warning: {results['notes']}")
        return results
    
    # 1. Latency benchmark (5 runs on first sentence)
    print("\n[1/5] Latency benchmark...")
    try:
        temp_out = "temp_latency.wav"
        latency_results = measure_latency(pipeline, first_sentence, reference_audio, temp_out, num_runs=5)
        if latency_results["error"]:
            results["notes"] += f"Latency error: {latency_results['error']}. "
            print(f"  Error: {latency_results['error']}")
        else:
            results["latency_ms"] = latency_results["median_latency_ms"]
            print(f"  Median latency: {results['latency_ms']:.2f} ms")
        
        # Cleanup
        if os.path.exists(temp_out):
            os.remove(temp_out)
    except Exception as e:
        results["notes"] += f"Latency benchmark failed: {str(e)}. "
        print(f"  Error: {str(e)}")
    
    # 2. RTF benchmark (on first sample)
    print("\n[2/5] RTF benchmark...")
    try:
        # Get generation time from pipeline result
        result = pipeline.synthesize(first_sentence, reference_audio, "temp_rtf.wav")
        if result["success"]:
            rtf_results = calculate_rtf(result["gen_time_sec"], "temp_rtf.wav")
            if rtf_results["error"]:
                results["notes"] += f"RTF error: {rtf_results['error']}. "
                print(f"  Error: {rtf_results['error']}")
            else:
                results["RTF"] = rtf_results["rtf"]
                print(f"  RTF: {results['RTF']:.4f}")
            
            # Cleanup
            if os.path.exists("temp_rtf.wav"):
                os.remove("temp_rtf.wav")
        else:
            results["notes"] += f"RTF synthesis failed: {result['error']}. "
            print(f"  Error: {result['error']}")
    except Exception as e:
        results["notes"] += f"RTF benchmark failed: {str(e)}. "
        print(f"  Error: {str(e)}")
    
    # 3. Speaker similarity (average over all samples)
    print("\n[3/5] Speaker similarity...")
    try:
        similarities = []
        for sample_file in sample_files[:5]:  # Limit to first 5 samples
            sim_result = compute_similarity(reference_audio, str(sample_file))
            if sim_result["error"]:
                print(f"  {sample_file.name}: Error - {sim_result['error']}")
            else:
                similarities.append(sim_result["cosine_similarity"])
                print(f"  {sample_file.name}: {sim_result['cosine_similarity']:.4f}")
        
        if similarities:
            results["speaker_cosine_sim"] = sum(similarities) / len(similarities)
            print(f"  Average similarity: {results['speaker_cosine_sim']:.4f}")
        else:
            results["notes"] += "No speaker similarity measurements succeeded. "
    except Exception as e:
        results["notes"] += f"Speaker similarity failed: {str(e)}. "
        print(f"  Error: {str(e)}")
    
    # 4. Round-trip WER (on first sample)
    print("\n[4/5] Round-trip WER...")
    try:
        transcribe_result = transcribe_audio(str(first_sample), language=language)
        if transcribe_result["error"]:
            results["notes"] += f"Transcription error: {transcribe_result['error']}. "
            print(f"  Error: {transcribe_result['error']}")
        else:
            wer_result = compute_wer(first_sentence, transcribe_result["text"])
            if wer_result["error"]:
                results["notes"] += f"WER error: {wer_result['error']}. "
                print(f"  Error: {wer_result['error']}")
            else:
                results["WER"] = wer_result["wer"]
                print(f"  WER: {results['WER']:.4f} ({results['WER']*100:.2f}%)")
    except Exception as e:
        results["notes"] += f"WER benchmark failed: {str(e)}. "
        print(f"  Error: {str(e)}")
    
    # 5. MOS (check if ratings exist)
    print("\n[5/5] MOS...")
    mos_csv = f"results/mos_raw_{language}_{model}.csv"
    if os.path.exists(mos_csv):
        mos_result = compute_average_mos(mos_csv)
        if mos_result["error"]:
            results["notes"] += f"MOS error: {mos_result['error']}. "
            print(f"  Error: {mos_result['error']}")
        else:
            results["MOS"] = mos_result["average_mos"]
            print(f"  Average MOS: {results['MOS']:.2f} ({mos_result['num_ratings']} ratings)")
    else:
        results["notes"] += f"MOS ratings not found at {mos_csv}. Run mos_collect.py first. "
        print(f"  Info: Run mos_collect.py to collect ratings")
    
    # 6. Cross-language consistency check
    print("\n[6/6] Cross-language consistency check...")
    targets = {
        "MOS": 4.0,
        "speaker_cosine_sim": 0.75,
        "latency_ms": 2000,  # batch target
        "RTF": 0.5,
        "WER": 0.10
    }
    
    missed_targets = []
    for metric, target in targets.items():
        value = results.get(metric)
        if value is not None:
            if metric == "latency_ms":
                if value > target:
                    missed_targets.append(metric)
            elif metric == "RTF" or metric == "WER":
                if value > target:
                    missed_targets.append(metric)
            else:
                if value < target:
                    missed_targets.append(metric)
    
    if missed_targets:
        results["notes"] += f"Missed targets: {', '.join(missed_targets)}. "
        print(f"  Missed targets: {', '.join(missed_targets)}")
    else:
        print(f"  All targets met (for metrics with data)")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run all evaluations")
    parser.add_argument("--hardware", default="T4", help="Hardware identifier")
    parser.add_argument("--ref", default="data/reference_audio/me.wav", help="Reference audio path")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.ref):
        print(f"Error: Reference audio not found: {args.ref}")
        return
    
    # Define all pipelines to evaluate
    pipelines = [
        {
            "language": "en",
            "model": "chatterbox",
            "pipeline_module": "pipelines/english/run_chatterbox.py",
            "samples_dir": "results/samples/english/chatterbox",
            "eval_sentences": "data/eval_sentences/english.txt"
        },
        {
            "language": "en",
            "model": "cosyvoice2",
            "pipeline_module": "pipelines/english/run_cosyvoice2.py",
            "samples_dir": "results/samples/english/cosyvoice2",
            "eval_sentences": "data/eval_sentences/english.txt"
        },
        {
            "language": "ar",
            "model": "xtts-v2",
            "pipeline_module": "pipelines/arabic/run_xtts.py",
            "samples_dir": "results/samples/arabic/xtts",
            "eval_sentences": "data/eval_sentences/arabic.txt"
        },
        {
            "language": "ar",
            "model": "mms-tts",
            "pipeline_module": "pipelines/arabic/run_mms_tts.py",
            "samples_dir": "results/samples/arabic/mms_tts",
            "eval_sentences": "data/eval_sentences/arabic.txt"
        },
        {
            "language": "hi",
            "model": "indic-parler-tts",
            "pipeline_module": "pipelines/hindi/run_indic_parler_tts.py",
            "samples_dir": "results/samples/hindi/indic_parler",
            "eval_sentences": "data/eval_sentences/hindi.txt"
        },
        {
            "language": "hi",
            "model": "xtts-v2",
            "pipeline_module": "pipelines/hindi/run_xtts_hindi.py",
            "samples_dir": "results/samples/hindi/xtts",
            "eval_sentences": "data/eval_sentences/hindi.txt"
        },
    ]
    
    # Run evaluations
    all_results = []
    for pipeline_config in pipelines:
        result = evaluate_pipeline(
            language=pipeline_config["language"],
            model=pipeline_config["model"],
            pipeline_module=pipeline_config["pipeline_module"],
            reference_audio=args.ref,
            samples_dir=pipeline_config["samples_dir"],
            eval_sentences_file=pipeline_config["eval_sentences"],
            hardware=args.hardware
        )
        all_results.append(result)
    
    # Save results to CSV
    df = pd.DataFrame(all_results)
    
    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)
    
    # Save to results.csv
    df.to_csv("results/results.csv", index=False)
    
    print(f"\n{'='*60}")
    print("Evaluation complete!")
    print(f"Results saved to results/results.csv")
    print(f"{'='*60}")
    
    # Print summary
    print("\nSummary:")
    print(df[["language", "model", "MOS", "speaker_cosine_sim", "latency_ms", "RTF", "WER"]].to_string(index=False))


if __name__ == "__main__":
    main()
