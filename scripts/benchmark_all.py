"""
Comprehensive benchmark runner for Vortex-Codec compression.

Compares Vortex-Codec against baseline compressors (LZMA, ZLIB, Gzip, Zstd)
across multiple datasets with detailed metrics including compression ratio,
compression time, and statistical significance.
"""

import subprocess
import sys
from pathlib import Path
import json
import time
import gzip
import lzma
import zlib
from typing import Dict, List, Tuple, Optional
import statistics


def compress_baseline(data: bytes, method: str) -> Tuple[bytes, float]:
    """Compress data using baseline methods and measure time."""
    start = time.perf_counter()
    
    if method == 'gzip':
        compressed = gzip.compress(data, compresslevel=6)
    elif method == 'zlib':
        compressed = zlib.compress(data, level=9)
    elif method == 'lzma':
        compressed = lzma.compress(data, preset=9)
    else:
        raise ValueError(f"Unknown compression method: {method}")
    
    elapsed = time.perf_counter() - start
    return compressed, elapsed


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """Calculate mean and standard deviation for timing measurements."""
    if len(values) < 2:
        return {'mean': values[0] if values else 0, 'std': 0, 'σ_SSR': 0}
    
    mean = statistics.mean(values)
    std = statistics.stdev(values)
    σ_SSR = std / mean if mean > 0 else 0
    
    return {'mean': mean, 'std': std, 'σ_SSR': σ_SSR}


def run_baseline_benchmark(data_path: Path, max_bytes: Optional[int] = None, 
                          num_runs: int = 3) -> Dict[str, Dict]:
    """Run baseline compression benchmarks with multiple runs for statistical accuracy."""
    
    # Read data
    with open(data_path, 'rb') as f:
        data = f.read(max_bytes) if max_bytes else f.read()
    
    original_size = len(data)
    baseline_size_mb = original_size / (1024 * 1024)
    
    results = {
        'baseline': {
            'size_mb': baseline_size_mb,
            'compression_ratio': 'N/A',
            'compression_time_s': 'N/A',
            'σ_SSR': 'N/A'
        }
    }
    
    # Test each compression method
    methods = {
        'LZMA(9)': 'lzma',
        'ZLIB(9)': 'zlib', 
        'Gzip(6)': 'gzip'
    }
    
    print(f"\n{'Compressor':<15} {'Size (MB)':<12} {'Ratio':<12} {'Time (s)':<15} {'σ_SSR':<8}")
    print("-" * 70)
    print(f"{'Baseline':<15} {baseline_size_mb:<12.2f} {'N/A':<12} {'N/A':<15} {'N/A':<8}")
    
    for method_name, method_type in methods.items():
        times = []
        compressed_size = None
        
        # Run multiple times for statistical accuracy
        for run in range(num_runs):
            try:
                compressed, elapsed = compress_baseline(data, method_type)
                times.append(elapsed)
                compressed_size = len(compressed)
            except Exception as e:
                print(f"Error with {method_name}: {e}")
                break
        
        if compressed_size and times:
            stats = calculate_statistics(times)
            compressed_mb = compressed_size / (1024 * 1024)
            ratio = original_size / compressed_size
            
            results[method_name] = {
                'size_mb': compressed_mb,
                'compression_ratio': ratio,
                'compression_time_s': stats['mean'],
                'σ_SSR': stats['σ_SSR']
            }
            
            print(f"{method_name:<15} {compressed_mb:<12.2f} {ratio:<12.2f} "
                  f"{stats['mean']:<15.2f} {stats['σ_SSR']:<8.2f}")
    
    return results


def run_vortex_benchmark(data_path: Path, model_path: Path, 
                        max_bytes: Optional[int] = None,
                        num_runs: int = 1) -> Dict[str, any]:
    """Run Vortex-Codec benchmark with timing."""
    
    # Read original data for size comparison
    with open(data_path, 'rb') as f:
        data = f.read(max_bytes) if max_bytes else f.read()
    original_size = len(data)
    
    times = []
    compressed_path = Path("temp_compressed.vxc")
    
    print(f"\n{'Vortex-Codec':<15} Running neural compression (this may take a few minutes)...")
    
    # Run multiple times for statistical accuracy (default 1 for neural due to speed)
    for run in range(num_runs):
        try:
            if num_runs > 1:
                print(f"  Run {run + 1}/{num_runs}...", end=" ", flush=True)
            
            start = time.perf_counter()
            
            cmd = [
                sys.executable,
                "bin/compress.py",
                str(data_path),  # positional argument
                "-o", str(compressed_path),
                "-m", str(model_path)
            ]
            
            if max_bytes:
                cmd.extend(["--max-bytes", str(max_bytes)])
            
            # Don't capture output so user can see progress
            result = subprocess.run(cmd, check=True)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            
            if num_runs > 1:
                print(f"done ({elapsed:.1f}s)")
            
        except subprocess.CalledProcessError as e:
            print(f"\nError running Vortex-Codec: {e}")
            if hasattr(e, 'stderr') and e.stderr:
                print(f"stderr: {e.stderr}")
            return None
    
    # Get compressed size
    if compressed_path.exists():
        compressed_size = compressed_path.stat().st_size
        compressed_path.unlink()  # Clean up
        
        stats = calculate_statistics(times)
        compressed_mb = compressed_size / (1024 * 1024)
        ratio = original_size / compressed_size
        
        # For single run, don't show σ_SSR
        sigma_display = f"{stats['σ_SSR']:.2f}" if num_runs > 1 else "N/A"
        
        print(f"{'Vortex-Codec':<15} {compressed_mb:<12.2f} {ratio:<12.2f} "
              f"{stats['mean']:<15.2f} {sigma_display:<8}")
        
        return {
            'size_mb': compressed_mb,
            'compression_ratio': ratio,
            'compression_time_s': stats['mean'],
            'σ_SSR': stats['σ_SSR'] if num_runs > 1 else 'N/A'
        }
    
    return None


def run_benchmarks(model_path: Optional[str] = None, vortex_runs: int = 1):
    """Run comprehensive benchmarks on multiple ATLAS datasets."""
    experiments_dir = Path("experiments/atlas_experiment")
    
    if not experiments_dir.exists():
        print("Error: experiments/atlas_experiment/ directory not found.")
        sys.exit(1)
    
    # Define test datasets (similar to Image 2's multiple dataset approach)
    datasets = {
        'ATLAS_200M': {
            'path': experiments_dir / 'atlas_200m.bin',
            'max_bytes': None,
            'description': 'ATLAS jet detector data (200MB - training data for reference)'
        },
        'ATLAS_10M': {
            'path': experiments_dir / 'atlas_10m.bin',
            'max_bytes': None,
            'description': 'ATLAS jet detector data (10MB sample)'
        },
        'ATLAS_25M': {
            'path': experiments_dir / 'atlas_25m.bin',
            'max_bytes': None,
            'description': 'ATLAS jet detector data (25MB sample)'
        },
        'ATLAS_50M': {
            'path': experiments_dir / 'atlas_50m.bin',
            'max_bytes': None,
            'description': 'ATLAS jet detector data (50MB sample)'
        },
        'ATLAS_Full': {
            'path': experiments_dir / 'atlas.bin',
            'max_bytes': 100_000_000,
            'description': 'ATLAS jet detector data (100MB subset)'
        },
    }
    
    print("=" * 80)
    print("COMPREHENSIVE COMPRESSION BENCHMARKS - Vortex-Codec vs Baseline Compressors")
    print("=" * 80)
    if model_path:
        print(f"Model: {model_path}")
        print(f"Vortex-Codec runs: {vortex_runs}x (use --runs to change)")
    print(f"Baseline compressors: LZMA(9), ZLIB(9), Gzip(6) - each run 3x")
    print(f"Metrics: Size (MB), Compression Ratio, Time (s), σ_SSR (timing variance)")
    print("=" * 80)
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    all_results = {}
    
    for dataset_name, dataset_info in datasets.items():
        data_path = dataset_info['path']
        
        if not data_path.exists():
            print(f"\n⚠ Skipping {dataset_name}: file not found at {data_path}")
            continue
        
        file_size = data_path.stat().st_size
        
        print(f"\n\n{'='*80}")
        print(f"Dataset: {dataset_name}")
        print(f"Description: {dataset_info['description']}")
        print(f"Path: {data_path}")
        print(f"Size: {file_size:,} bytes ({file_size/1024/1024:.2f} MB)")
        print('='*80)
        
        # Run baseline benchmarks
        baseline_results = run_baseline_benchmark(
            data_path, 
            max_bytes=dataset_info['max_bytes'],
            num_runs=3
        )
        
        # Run Vortex-Codec if model provided
        if model_path:
            vortex_results = run_vortex_benchmark(
                data_path,
                Path(model_path),
                max_bytes=dataset_info['max_bytes'],
                num_runs=vortex_runs  # Neural compression is slow, default=1
            )
            
            if vortex_results:
                baseline_results['Vortex-Codec'] = vortex_results
        
        all_results[dataset_name] = baseline_results
        
        # Save individual results
        output_file = results_dir / f"{dataset_name.lower()}_comparison.json"
        with open(output_file, 'w') as f:
            json.dump(baseline_results, f, indent=2)
    
    # Save combined results
    summary_file = results_dir / "compression_benchmark_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)
    print(f"\n✓ Results saved to: {results_dir}/")
    print(f"✓ Summary: {summary_file}")
    
    if not model_path:
        print("\n⚠ No model provided - only baseline compressors tested")
        print("\nTo include Vortex-Codec in benchmarks:")
        print("  python scripts/benchmark_all.py --model model/best_model.pt")
    
    print("\nNext steps:")
    print("  • Train model: python train_example.py --data experiments/atlas_experiment/atlas_200m.bin")
    print("  • Compress: python bin/compress.py <input-file> -o <output.vxc> -m <model.pt>")
    print("  • Decompress: python bin/decompress.py <input.vxc> -o <output-file> -m <model.pt>")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Comprehensive benchmark comparing Vortex-Codec against baseline compressors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Test baseline compressors only:
  python scripts/benchmark_all.py
  
  # Include Vortex-Codec comparison:
  python scripts/benchmark_all.py --model model/best_model.pt
        """
    )
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained Vortex-Codec model (optional - will run baselines only if not provided)')
    parser.add_argument('--runs', type=int, default=1,
                       help='Number of runs for Vortex-Codec compression (default: 1, increase for better statistics but slower)')
    
    args = parser.parse_args()
    run_benchmarks(model_path=args.model, vortex_runs=args.runs)