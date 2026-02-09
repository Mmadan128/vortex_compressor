"""
Evaluation and benchmarking script for Vortex-Codec.

Compares compression performance against standard codecs (Gzip, Zstd)
and provides detailed metrics on model performance.
"""

import torch
import argparse
import gzip
import time
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm

try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False

from vortex.core import VortexCodec
from vortex.io import ByteDataset
from vortex.utils.metrics import compute_bpd, evaluate_compression_performance
import yaml


def benchmark_gzip(data: bytes, level: int = 6) -> dict:
    """Benchmark Gzip compression."""
    start = time.time()
    compressed = gzip.compress(data, compresslevel=level)
    compress_time = time.time() - start
    
    start = time.time()
    decompressed = gzip.decompress(compressed)
    decompress_time = time.time() - start
    
    assert decompressed == data, "Gzip decompression failed"
    
    return {
        'name': f'Gzip (level {level})',
        'original_size': len(data),
        'compressed_size': len(compressed),
        'ratio': len(compressed) / len(data),
        'factor': len(data) / len(compressed),
        'savings_pct': (1 - len(compressed) / len(data)) * 100,
        'bpd': (len(compressed) * 8) / len(data),
        'compress_time': compress_time,
        'decompress_time': decompress_time,
        'compress_mbps': (len(data) / 1024 / 1024) / compress_time,
        'decompress_mbps': (len(data) / 1024 / 1024) / decompress_time
    }


def benchmark_zstd(data: bytes, level: int = 3) -> dict:
    """Benchmark Zstandard compression."""
    if not ZSTD_AVAILABLE:
        return None
    
    cctx = zstd.ZstdCompressor(level=level)
    dctx = zstd.ZstdDecompressor()
    
    start = time.time()
    compressed = cctx.compress(data)
    compress_time = time.time() - start
    
    start = time.time()
    decompressed = dctx.decompress(compressed)
    decompress_time = time.time() - start
    
    assert decompressed == data, "Zstd decompression failed"
    
    return {
        'name': f'Zstd (level {level})',
        'original_size': len(data),
        'compressed_size': len(compressed),
        'ratio': len(compressed) / len(data),
        'factor': len(data) / len(compressed),
        'savings_pct': (1 - len(compressed) / len(data)) * 100,
        'bpd': (len(compressed) * 8) / len(data),
        'compress_time': compress_time,
        'decompress_time': decompress_time,
        'compress_mbps': (len(data) / 1024 / 1024) / compress_time,
        'decompress_mbps': (len(data) / 1024 / 1024) / decompress_time
    }


def evaluate_model_bpd(model, data_path: str, device, max_bytes: int = None):
    """Evaluate model BPD on test data."""
    print("\nEvaluating model on test data...")
    
    dataset = ByteDataset(
        file_path=data_path,
        window_size=512,
        stride=512,
        max_bytes=max_bytes
    )
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=2
    )
    
    model.eval()
    total_bpd = 0
    num_batches = 0
    
    # Initialize compressed memories for proper compressive transformer evaluation
    compressed_memories = None
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Computing BPD"):
            batch = batch.to(device)
            
            # Maintain memory state across batches
            logits, compressed_memories = model(batch, compressed_memories=compressed_memories)
            
            bpd = compute_bpd(logits[:, :-1], batch[:, 1:])
            total_bpd += bpd
            num_batches += 1
    
    avg_bpd = total_bpd / num_batches
    return avg_bpd


def run_benchmark(args):
    """Run complete benchmark suite."""
    print("=" * 70)
    print("Vortex-Codec Compression Benchmark")
    print("=" * 70)
    
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        return
    
    with open(data_path, 'rb') as f:
        if args.max_bytes:
            raw_data = f.read(args.max_bytes)
        else:
            raw_data = f.read()
    
    print(f"\nDataset: {data_path.name}")
    print(f"Size: {len(raw_data):,} bytes ({len(raw_data)/1024/1024:.2f} MB)")
    
    results = {}
    
    print("\n" + "=" * 70)
    print("Baseline Compression Benchmarks")
    print("=" * 70)
    
    print("\nBenchmarking Gzip...")
    gzip_result = benchmark_gzip(raw_data, level=6)
    results['gzip'] = gzip_result
    print(f"  Ratio: {gzip_result['ratio']:.4f} ({gzip_result['factor']:.2f}x compression)")
    print(f"  BPD: {gzip_result['bpd']:.4f}")
    print(f"  Speed: {gzip_result['compress_mbps']:.2f} MB/s (compress), "
          f"{gzip_result['decompress_mbps']:.2f} MB/s (decompress)")
    
    if ZSTD_AVAILABLE:
        print("\nBenchmarking Zstd...")
        zstd_result = benchmark_zstd(raw_data, level=3)
        results['zstd'] = zstd_result
        print(f"  Ratio: {zstd_result['ratio']:.4f} ({zstd_result['factor']:.2f}x compression)")
        print(f"  BPD: {zstd_result['bpd']:.4f}")
        print(f"  Speed: {zstd_result['compress_mbps']:.2f} MB/s (compress), "
              f"{zstd_result['decompress_mbps']:.2f} MB/s (decompress)")
    
    if args.model:
        print("\n" + "=" * 70)
        print("Neural Compression (Vortex-Codec)")
        print("=" * 70)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"\nDevice: {device}")
        
        print(f"Loading model from {args.model}...")
        
        if args.config:
            with open(args.config) as f:
                config = yaml.safe_load(f)
        else:
            config = {
                'model': {'d_model': 256, 'n_layers': 6, 'n_heads': 8, 'd_ff': 1024, 
                         'dropout': 0.1, 'vocab_size': 256},
                'compressive_memory': {'window_size': 512, 'compression_rate': 4}
            }
        
        model = VortexCodec(
            **config['model'],
            **config['compressive_memory']
        ).to(device)
        
        if args.model.endswith('.pt'):
            checkpoint = torch.load(args.model, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"  Loaded from checkpoint (epoch {checkpoint.get('epoch', '?')})")
                if 'val_bpd' in checkpoint:
                    print(f"  Checkpoint validation BPD: {checkpoint['val_bpd']:.4f}")
            else:
                model.load_state_dict(checkpoint)
        
        model_bpd = evaluate_model_bpd(
            model, 
            data_path, 
            device, 
            max_bytes=args.max_bytes
        )
        
        results['vortex'] = {
            'name': 'Vortex-Codec (Neural)',
            'model_bpd': model_bpd,
            'theoretical_ratio': model_bpd / 8,
            'theoretical_factor': 8 / model_bpd,
            'note': 'BPD only (actual compression requires arithmetic coding)'
        }
        
        print(f"\nModel BPD: {model_bpd:.4f}")
        print(f"Theoretical compression ratio: {model_bpd/8:.4f}")
        print(f"Theoretical compression factor: {8/model_bpd:.2f}x")
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    
    print(f"\n{'Method':<25} {'BPD':<10} {'Ratio':<10} {'Factor':<10}")
    print("-" * 70)
    
    if 'gzip' in results:
        r = results['gzip']
        print(f"{r['name']:<25} {r['bpd']:<10.4f} {r['ratio']:<10.4f} {r['factor']:<10.2f}x")
    
    if 'zstd' in results:
        r = results['zstd']
        print(f"{r['name']:<25} {r['bpd']:<10.4f} {r['ratio']:<10.4f} {r['factor']:<10.2f}x")
    
    if 'vortex' in results:
        r = results['vortex']
        print(f"{r['name']:<25} {r['model_bpd']:<10.4f} {r['theoretical_ratio']:<10.4f} {r['theoretical_factor']:<10.2f}x")
    
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Evaluate and benchmark Vortex-Codec")
    parser.add_argument('--data', type=str, required=True,
                       help='Path to test data file')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to model config (if not in checkpoint)')
    parser.add_argument('--max-bytes', type=int, default=None,
                       help='Maximum bytes to evaluate (for large files)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for JSON results')
    
    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
