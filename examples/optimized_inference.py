#!/usr/bin/env python3
"""
Example script demonstrating optimized inference with Flash Attention 3.

This script shows how to:
1. Load a trained model
2. Apply Flash Attention + Mixed Precision + torch.compile()
3. Benchmark performance
4. Run optimized compression

Usage:
    python examples/optimized_inference.py \
        --model checkpoints/atlas_10m/best_model.pt \
        --input experiments/atlas_experiment/atlas_10m.bin \
        --output test_optimized.vxc
"""

import argparse
import time
import torch
import numpy as np
from pathlib import Path

from vortex.core import VortexCodec
from vortex.utils import optimize_for_inference, benchmark_inference


def main():
    parser = argparse.ArgumentParser(description='Optimized Vortex Codec Inference')
    parser.add_argument('--model', type=str, default='model/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                        help='Input file to compress')
    parser.add_argument('--output', type=str, default='compressed.vxc',
                        help='Output compressed file')
    parser.add_argument('--chunk-size', type=int, default=2048,
                        help='Chunk size for compression (larger = faster)')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                        choices=['float32', 'float16', 'bfloat16'],
                        help='Precision for inference')
    parser.add_argument('--compile', action='store_true',
                        help='Use torch.compile() for extra speedup')
    parser.add_argument('--no-flash', action='store_true',
                        help='Disable Flash Attention (for comparison)')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run benchmarks before compression')
    
    args = parser.parse_args()
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*70}")
    print(f"VORTEX CODEC - OPTIMIZED INFERENCE")
    print(f"{'='*70}")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"{'='*70}\n")
    
    # 1. Create model with Flash Attention
    print("📦 Loading model...")
    use_flash = not args.no_flash
    model = VortexCodec(
        d_model=256,
        n_layers=6,
        n_heads=8,
        d_ff=1024,
        window_size=512,
        use_flash_attention=use_flash,
        flash_backend='auto'
    )
    
    # Load checkpoint
    if Path(args.model).exists():
        checkpoint = torch.load(args.model, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"✓ Loaded checkpoint: {args.model}")
    else:
        print(f"⚠ Checkpoint not found: {args.model}")
        print(f"  Using randomly initialized model for demonstration")
    
    # 2. Apply optimizations
    print("\n⚡ Applying optimizations...")
    
    # Mixed precision
    dtype_map = {
        'float32': None,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16
    }
    target_dtype = dtype_map[args.dtype]
    
    # Compile mode
    compile_mode = 'reduce-overhead' if args.compile else None
    
    model = optimize_for_inference(
        model,
        dtype=target_dtype,
        compile_mode=compile_mode
    )
    
    model = model.to(device)
    
    # CUDA optimizations
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
        print("✓ CUDA optimizations enabled")
    
    print(f"✓ Flash Attention: {'Enabled' if use_flash else 'Disabled'}")
    print(f"✓ Precision: {args.dtype}")
    print(f"✓ torch.compile(): {'Enabled' if args.compile else 'Disabled'}")
    
    # 3. Benchmark (optional)
    if args.benchmark:
        print(f"\n📊 Running benchmarks...")
        sample_data = torch.randint(0, 256, (1, args.chunk_size), device=device)
        
        # Warmup for compiled model
        if args.compile:
            print("  Warming up compiled model (this may take 10-30 seconds)...")
            with torch.no_grad():
                for _ in range(3):
                    _ = model(sample_data)
        
        stats = benchmark_inference(
            model,
            sample_data,
            num_warmup=5,
            num_iterations=50
        )
        
        print(f"\n{'='*70}")
        print(f"BENCHMARK RESULTS:")
        print(f"{'='*70}")
        print(f"Average Latency:  {stats['avg_latency_ms']:.2f}ms (±{stats['std_latency_ms']:.2f}ms)")
        print(f"Min Latency:      {stats['min_latency_ms']:.2f}ms")
        print(f"Max Latency:      {stats['max_latency_ms']:.2f}ms")
        print(f"Throughput:       {stats['throughput_samples_per_sec']:.1f} samples/sec")
        if device.type == 'cuda':
            print(f"GPU Memory:       {stats['memory_allocated_mb']:.1f}MB allocated")
            print(f"                  {stats['memory_reserved_mb']:.1f}MB reserved")
        print(f"{'='*70}\n")
    
    # 4. Load input data
    print(f"📂 Loading input file: {args.input}")
    if not Path(args.input).exists():
        print(f"✗ Input file not found: {args.input}")
        return
    
    data = np.fromfile(args.input, dtype=np.uint8)
    print(f"✓ Loaded {len(data):,} bytes")
    
    # 5. Compress with optimizations
    print(f"\n🗜️  Compressing with optimized settings...")
    print(f"  Chunk size: {args.chunk_size}")
    
    start_time = time.time()
    
    with torch.no_grad():
        compressed = model.compress(
            data,
            chunk_size=args.chunk_size,
            show_progress=True
        )
    
    elapsed = time.time() - start_time
    
    # 6. Save compressed output
    with open(args.output, 'wb') as f:
        f.write(compressed)
    
    # 7. Results
    print(f"\n{'='*70}")
    print(f"COMPRESSION RESULTS:")
    print(f"{'='*70}")
    print(f"Original size:      {len(data):,} bytes")
    print(f"Compressed size:    {len(compressed):,} bytes")
    print(f"Compression ratio:  {len(data)/len(compressed):.3f}x")
    print(f"Space saved:        {(1 - len(compressed)/len(data))*100:.1f}%")
    print(f"Time elapsed:       {elapsed:.2f}s")
    print(f"Throughput:         {len(data)/elapsed/1024/1024:.2f} MB/s")
    print(f"Output saved to:    {args.output}")
    print(f"{'='*70}\n")
    
    # 8. Performance tips
    if not use_flash:
        print("💡 TIP: Enable Flash Attention (remove --no-flash) for 40-60% speedup!")
    if args.dtype == 'float32':
        print("💡 TIP: Use --dtype bfloat16 for 30-50% speedup!")
    if not args.compile:
        print("💡 TIP: Use --compile for additional 10-30% speedup!")
    if args.chunk_size < 2048:
        print("💡 TIP: Increase --chunk-size to 2048 or 4096 for better GPU utilization!")
    
    print("\n✓ Done!\n")


if __name__ == "__main__":
    main()
