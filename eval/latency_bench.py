"""
Latency benchmark for TTS pipelines.

Measures time to first audio chunk (streaming) or full clip (batch).
"""

import os
import sys
import time
import argparse
import statistics

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'pipelines', 'common'))

from audio_utils import load_audio, get_audio_duration


def measure_latency(pipeline, text: str, reference_audio: str, out_path: str, num_runs: int = 5) -> dict:
    """
    Measure synthesis latency over multiple runs.
    
    Args:
        pipeline: TTS pipeline instance
        text: Text to synthesize
        reference_audio: Path to reference audio
        out_path: Output path for audio
        num_runs: Number of runs to perform
    
    Returns:
        Dict with latency statistics
    """
    latencies = []
    
    for i in range(num_runs):
        result = pipeline.synthesize(text, reference_audio, out_path)
        
        if result["success"]:
            latencies.append(result["gen_time_sec"] * 1000)  # Convert to ms
        else:
            print(f"Run {i+1} failed: {result['error']}")
            return {
                "median_latency_ms": None,
                "mean_latency_ms": None,
                "min_latency_ms": None,
                "max_latency_ms": None,
                "std_latency_ms": None,
                "successful_runs": 0,
                "total_runs": num_runs,
                "error": result["error"]
            }
    
    if not latencies:
        return {
            "median_latency_ms": None,
            "mean_latency_ms": None,
            "min_latency_ms": None,
            "max_latency_ms": None,
            "std_latency_ms": None,
            "successful_runs": 0,
            "total_runs": num_runs,
            "error": "No successful runs"
        }
    
    return {
        "median_latency_ms": statistics.median(latencies),
        "mean_latency_ms": statistics.mean(latencies),
        "min_latency_ms": min(latencies),
        "max_latency_ms": max(latencies),
        "std_latency_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
        "successful_runs": len(latencies),
        "total_runs": num_runs,
        "error": None
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark TTS latency")
    parser.add_argument("--pipeline", required=True, help="Pipeline module path (e.g., pipelines.english.run_chatterbox)")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--ref", required=True, help="Reference audio path")
    parser.add_argument("--out", required=True, help="Output path")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs")
    parser.add_argument("--hardware", default="T4", help="Hardware identifier")
    
    args = parser.parse_args()
    
    # Import pipeline module
    module_path = args.pipeline.replace('.', '/')
    if not module_path.endswith('.py'):
        module_path += '.py'
    
    # Dynamic import
    import importlib.util
    spec = importlib.util.spec_from_file_location("pipeline_module", module_path)
    pipeline_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline_module)
    
    # Get pipeline class (assumes class name ends with Pipeline)
    pipeline_class = None
    for name in dir(pipeline_module):
        if name.endswith('Pipeline'):
            pipeline_class = getattr(pipeline_module, name)
            break
    
    if not pipeline_class:
        print("Error: Could not find pipeline class")
        return
    
    # Initialize and load pipeline
    pipeline = pipeline_class()
    pipeline.load(args.ref)
    
    # Measure latency
    print(f"Measuring latency over {args.runs} runs...")
    print(f"Text: {args.text}")
    
    results = measure_latency(pipeline, args.text, args.ref, args.out, args.runs)
    
    if results["error"]:
        print(f"Error: {results['error']}")
    else:
        print(f"\nLatency Results ({args.hardware}):")
        print(f"  Median: {results['median_latency_ms']:.2f} ms")
        print(f"  Mean:   {results['mean_latency_ms']:.2f} ms")
        print(f"  Min:    {results['min_latency_ms']:.2f} ms")
        print(f"  Max:    {results['max_latency_ms']:.2f} ms")
        print(f"  Std:    {results['std_latency_ms']:.2f} ms")
        print(f"  Runs:   {results['successful_runs']}/{results['total_runs']}")
        
        # Check targets
        target_streaming = 500  # ms
        target_batch = 2000  # ms
        
        if results["median_latency_ms"] < target_streaming:
            print(f"  ✓ Meets streaming target (<{target_streaming}ms)")
        elif results["median_latency_ms"] < target_batch:
            print(f"  ✓ Meets batch target (<{target_batch}ms)")
        else:
            print(f"  ✗ Does not meet latency targets")


if __name__ == "__main__":
    main()
