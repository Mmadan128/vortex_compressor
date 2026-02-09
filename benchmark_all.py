"""
Quick benchmark runner for comparing codecs on all test datasets.

Runs baseline compression benchmarks (Gzip, Zstd) on all generated datasets
without requiring a trained model.
"""

import subprocess
import sys
from pathlib import Path
import json


def run_benchmarks():
    """Run benchmarks on all test datasets."""
    data_dir = Path("data")
    
    if not data_dir.exists():
        print("Error: data/ directory not found. Run generate_test_data.py first.")
        sys.exit(1)
    
    manifest_file = data_dir / "manifest.json"
    if not manifest_file.exists():
        print("Error: manifest.json not found. Run generate_test_data.py first.")
        sys.exit(1)
    
    with open(manifest_file) as f:
        manifest = json.load(f)
    
    print("=" * 70)
    print("Running Compression Benchmarks on All Datasets")
    print("=" * 70)
    
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    
    for dataset_name, dataset_info in manifest['datasets'].items():
        data_path = dataset_info['path']
        
        print(f"\n\n{'='*70}")
        print(f"Dataset: {dataset_name}")
        print(f"Path: {data_path}")
        print(f"Size: {dataset_info['size_bytes']:,} bytes")
        print(f"Type: {dataset_info['type']}")
        print('='*70)
        
        output_file = results_dir / f"{dataset_name}_results.json"
        
        cmd = [
            sys.executable,
            "evaluate.py",
            "--data", data_path,
            "--output", str(output_file),
            "--max-bytes", "10000000"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=False, text=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running benchmark on {dataset_name}: {e}")
            continue
    
    print("\n" + "=" * 70)
    print("All Benchmarks Complete")
    print("=" * 70)
    print(f"\nResults saved to: {results_dir}/")
    print("\nTo train a model on a dataset:")
    print("  python train_example.py --data data/<dataset>.bin --epochs 10")


if __name__ == "__main__":
    run_benchmarks()
