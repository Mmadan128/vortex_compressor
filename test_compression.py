"""
Test script for full compression pipeline.

Demonstrates actual compression with trained model, measuring real
compression ratios and comparing against theoretical BPD estimates.
"""

import torch
import argparse
from pathlib import Path
import gzip
import time

try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False

from vortex.core import VortexCodec
from vortex.io import ByteDataset
from vortex.utils.metrics import compute_bpd
import yaml


def load_model(model_path: str, config_path: str = None, device=None):
    """Load trained model from checkpoint."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(model_path, map_location=device)
    
    if config_path:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    elif 'config' in checkpoint:
        config = checkpoint['config']
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
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    return model, config, checkpoint


def test_compression(data_path: str, model_path: str, max_bytes: int = None):
    """Test compression on a file."""
    print("=" * 70)
    print("Vortex-Codec Compression Test")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    data_path = Path(data_path)
    with open(data_path, 'rb') as f:
        raw_data = f.read(max_bytes) if max_bytes else f.read()
    
    print(f"Test file: {data_path.name}")
    print(f"Size: {len(raw_data):,} bytes ({len(raw_data)/1024/1024:.2f} MB)\n")
    
    print("-" * 70)
    print("Baseline Codecs")
    print("-" * 70)
    
    print("\n[1/2] Gzip...")
    start = time.time()
    gzip_compressed = gzip.compress(raw_data, compresslevel=6)
    gzip_time = time.time() - start
    gzip_ratio = len(gzip_compressed) / len(raw_data)
    print(f"  Size: {len(gzip_compressed):,} bytes")
    print(f"  Ratio: {gzip_ratio:.4f} ({1/gzip_ratio:.2f}x compression)")
    print(f"  BPD: {(len(gzip_compressed) * 8) / len(raw_data):.4f}")
    print(f"  Time: {gzip_time:.2f}s ({len(raw_data)/1024/1024/gzip_time:.2f} MB/s)")
    
    if ZSTD_AVAILABLE:
        print("\n[2/2] Zstd...")
        cctx = zstd.ZstdCompressor(level=3)
        start = time.time()
        zstd_compressed = cctx.compress(raw_data)
        zstd_time = time.time() - start
        zstd_ratio = len(zstd_compressed) / len(raw_data)
        print(f"  Size: {len(zstd_compressed):,} bytes")
        print(f"  Ratio: {zstd_ratio:.4f} ({1/zstd_ratio:.2f}x compression)")
        print(f"  BPD: {(len(zstd_compressed) * 8) / len(raw_data):.4f}")
        print(f"  Time: {zstd_time:.2f}s ({len(raw_data)/1024/1024/zstd_time:.2f} MB/s)")
    
    print("\n" + "-" * 70)
    print("Neural Compression (Vortex-Codec)")
    print("-" * 70 + "\n")
    
    print("Loading model...")
    model, config, checkpoint = load_model(model_path, device=device)
    
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if 'val_bpd' in checkpoint:
        print(f"  Validation BPD: {checkpoint['val_bpd']:.4f}")
    
    print("\n[1/2] Measuring model BPD...")
    dataset = ByteDataset(
        file_path=data_path,
        window_size=512,
        stride=512,
        max_bytes=max_bytes
    )
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)
    
    model.eval()
    total_bpd = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits, _ = model(batch)
            bpd = compute_bpd(logits[:, :-1], batch[:, 1:])
            total_bpd += bpd
            num_batches += 1
    
    model_bpd = total_bpd / num_batches
    print(f"  Model BPD: {model_bpd:.4f}")
    print(f"  Theoretical ratio: {model_bpd/8:.4f} ({8/model_bpd:.2f}x)")
    
    print("\n[2/2] Actual compression with arithmetic coding...")
    chunk_sizes = [128, 256, 512]
    
    for chunk_size in chunk_sizes:
        print(f"\n  Chunk size: {chunk_size}")
        try:
            start = time.time()
            compressed = model.compress(raw_data, chunk_size=chunk_size, show_progress=False)
            compress_time = time.time() - start
            
            actual_ratio = len(compressed) / len(raw_data)
            actual_bpd = (len(compressed) * 8) / len(raw_data)
            
            print(f"    Compressed size: {len(compressed):,} bytes")
            print(f"    Actual ratio: {actual_ratio:.4f} ({1/actual_ratio:.2f}x)")
            print(f"    Actual BPD: {actual_bpd:.4f}")
            print(f"    Time: {compress_time:.2f}s ({len(raw_data)/1024/1024/compress_time:.2f} MB/s)")
            print(f"    Overhead: {(actual_bpd - model_bpd):.4f} BPD")
            
        except Exception as e:
            print(f"    Error: {e}")
    
    print("\n" + "=" * 70)
    print("Compression Summary")
    print("=" * 70)
    print(f"\n{'Method':<30} {'BPD':<12} {'Ratio':<12} {'Factor':<10}")
    print("-" * 70)
    print(f"{'Gzip (level 6)':<30} {(len(gzip_compressed)*8)/len(raw_data):<12.4f} {gzip_ratio:<12.4f} {1/gzip_ratio:<10.2f}x")
    if ZSTD_AVAILABLE:
        print(f"{'Zstd (level 3)':<30} {(len(zstd_compressed)*8)/len(raw_data):<12.4f} {zstd_ratio:<12.4f} {1/zstd_ratio:<10.2f}x")
    print(f"{'Vortex (theoretical)':<30} {model_bpd:<12.4f} {model_bpd/8:<12.4f} {8/model_bpd:<10.2f}x")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Test Vortex-Codec compression")
    parser.add_argument('--data', type=str, required=True,
                       help='Path to test data file')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--max-bytes', type=int, default=1000000,
                       help='Maximum bytes to compress (default: 1MB)')
    
    args = parser.parse_args()
    
    test_compression(args.data, args.model, args.max_bytes)


if __name__ == "__main__":
    main()
