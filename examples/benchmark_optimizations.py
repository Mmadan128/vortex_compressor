#!/usr/bin/env python3
"""
Compare inference performance with different optimization strategies.

This script benchmarks:
1. Baseline (FP32, standard attention)
2. Flash Attention only
3. Flash Attention + BFloat16
4. Flash Attention + BFloat16 + torch.compile()

Usage:
    python examples/benchmark_optimizations.py --model model/best_model.pt
"""

import argparse
import torch
import numpy as np
from pathlib import Path

from vortex.core import VortexCodec
from vortex.utils import optimize_for_inference, benchmark_inference


def run_benchmark(config_name, model_kwargs, optimize_kwargs, device):
    """Run benchmark for a specific configuration."""
    
    print(f"\n{'='*70}")
    print(f"Testing: {config_name}")
    print(f"{'='*70}")
    
    # Create model
    model = VortexCodec(**model_kwargs)
    
    # Apply optimizations
    if optimize_kwargs:
        model = optimize_for_inference(model, **optimize_kwargs)
    
    model = model.to(device)
    model.eval()
    
    # Create sample data
    sample_data = torch.randint(0, 256, (1, 2048), device=device)
    
    # Warmup (especially important for compiled models)
    print("Warming up...")
    with torch.no_grad():
        for _ in range(5):
            _ = model(sample_data)
    
    # Benchmark
    print("Benchmarking...")
    stats = benchmark_inference(
        model,
        sample_data,
        num_warmup=5,
        num_iterations=50
    )
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Benchmark Vortex Codec Optimizations')
    parser.add_argument('--model', type=str, default='model/best_model.pt',
                        help='Path to model checkpoint (optional)')
    args = parser.parse_args()
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*70}")
    print(f"VORTEX CODEC - OPTIMIZATION BENCHMARK")
    print(f"{'='*70}")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    print(f"PyTorch: {torch.__version__}")
    print(f"{'='*70}")
    
    # Load checkpoint if available
    checkpoint = None
    if Path(args.model).exists():
        checkpoint = torch.load(args.model, map_location='cpu')
        print(f"✓ Loaded checkpoint: {args.model}\n")
    else:
        print(f"⚠ No checkpoint found, using random weights for demonstration\n")
    
    # Enable CUDA optimizations
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')
    
    # Test configurations
    configurations = [
        {
            'name': '1. Baseline (FP32, Standard Attention)',
            'model_kwargs': {
                'd_model': 256,
                'n_layers': 6,
                'n_heads': 8,
                'use_flash_attention': False
            },
            'optimize_kwargs': None
        },
        {
            'name': '2. Flash Attention Only',
            'model_kwargs': {
                'd_model': 256,
                'n_layers': 6,
                'n_heads': 8,
                'use_flash_attention': True,
                'flash_backend': 'auto'
            },
            'optimize_kwargs': None
        },
        {
            'name': '3. Flash Attention + BFloat16',
            'model_kwargs': {
                'd_model': 256,
                'n_layers': 6,
                'n_heads': 8,
                'use_flash_attention': True,
                'flash_backend': 'auto'
            },
            'optimize_kwargs': {
                'dtype': torch.bfloat16 if device.type == 'cuda' else None
            }
        },
        {
            'name': '4. Flash Attention + BFloat16 + torch.compile()',
            'model_kwargs': {
                'd_model': 256,
                'n_layers': 6,
                'n_heads': 8,
                'use_flash_attention': True,
                'flash_backend': 'auto'
            },
            'optimize_kwargs': {
                'dtype': torch.bfloat16 if device.type == 'cuda' else None,
                'compile_mode': 'reduce-overhead'
            }
        }
    ]
    
    # Run benchmarks
    results = []
    
    for config in configurations:
        try:
            stats = run_benchmark(
                config['name'],
                config['model_kwargs'],
                config['optimize_kwargs'],
                device
            )
            results.append({
                'name': config['name'],
                'stats': stats
            })
        except Exception as e:
            print(f"✗ Failed: {e}")
            results.append({
                'name': config['name'],
                'stats': None
            })
    
    # Print comparison table
    print(f"\n{'='*70}")
    print(f"COMPARISON TABLE")
    print(f"{'='*70}\n")
    
    baseline_latency = results[0]['stats']['avg_latency_ms'] if results[0]['stats'] else None
    
    print(f"{'Configuration':<45} {'Latency':<12} {'Speedup':<10} {'Memory'}")
    print(f"{'-'*45} {'-'*12} {'-'*10} {'-'*15}")
    
    for result in results:
        name = result['name'].split('. ')[1] if '. ' in result['name'] else result['name']
        
        if result['stats']:
            latency = result['stats']['avg_latency_ms']
            memory = result['stats']['memory_allocated_mb']
            
            speedup = baseline_latency / latency if baseline_latency else 1.0
            
            print(f"{name:<45} {latency:>8.2f}ms   {speedup:>5.2f}x     {memory:>6.1f}MB")
        else:
            print(f"{name:<45} {'FAILED':<12} {'-':<10} {'-'}")
    
    print(f"{'-'*70}\n")
    
    # Best configuration
    valid_results = [r for r in results if r['stats']]
    if valid_results:
        best_result = min(valid_results, key=lambda x: x['stats']['avg_latency_ms'])
        best_name = best_result['name'].split('. ')[1] if '. ' in best_result['name'] else best_result['name']
        best_latency = best_result['stats']['avg_latency_ms']
        best_speedup = baseline_latency / best_latency if baseline_latency else 1.0
        
        print(f"🏆 FASTEST CONFIGURATION: {best_name}")
        print(f"   Latency: {best_latency:.2f}ms")
        print(f"   Speedup: {best_speedup:.2f}x faster than baseline")
        print(f"   Memory:  {best_result['stats']['memory_allocated_mb']:.1f}MB\n")
    
    # Recommendations
    print(f"{'='*70}")
    print(f"RECOMMENDATIONS")
    print(f"{'='*70}\n")
    
    print("For production inference, use:")
    print("""
from vortex.core import VortexCodec
from vortex.utils import optimize_for_inference

model = VortexCodec(
    d_model=256,
    n_layers=6,
    n_heads=8,
    use_flash_attention=True,
    flash_backend='auto'
)

model.load_state_dict(torch.load('model.pt'))

model = optimize_for_inference(
    model,
    dtype=torch.bfloat16,
    compile_mode='reduce-overhead'
)

model = model.cuda()
""")
    
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
