"""Compress files with Vortex."""

import argparse
import sys
from pathlib import Path
import yaml

try:
    import torch
    from vortex.core import VortexCodec
except ImportError as e:
    print(f"Error: Required packages not installed: {e}")
    print("Install with: pip install -r requirements.txt")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Compress binary data using Vortex-Codec",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compress with trained model
  python -m bin.compress data.bin -o data.vortex -m checkpoints/model.pt
  
  # Compress with custom chunk size
  python -m bin.compress data.bin -o data.vortex -m model.pt --chunk-size 512
  
  # Compress only first 10 MB
  python -m bin.compress large.bin -o compressed.vortex -m model.pt --max-bytes 10000000
"""
    )
    
    parser.add_argument(
        "input",
        type=str,
        help="Path to input binary file"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Path to output compressed file (.vortex)"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pt)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (optional if checkpoint has config)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=256,
        help="Chunk size for compression (default: 256)"
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=None,
        help="Maximum bytes to compress (for testing on large files)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, default: auto-detect)"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)
    
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 70)
    print("Vortex-Codec Compression")
    print("=" * 70)
    print(f"Input: {input_path}")
    print(f"Output: {args.output}")
    print(f"Model: {model_path}")
    print(f"Device: {device}")
    print()
    
    print("Loading model...")
    checkpoint = torch.load(model_path, map_location=device)
    
    if 'config' in checkpoint and args.config is None:
        config = checkpoint['config']
        print("  Using config from checkpoint")
    elif args.config:
        with open(args.config) as f:
            config = yaml.safe_load(f)
        print(f"  Using config from {args.config}")
    else:
        print("  Using default config")
        config = {
            'model': {'d_model': 256, 'n_layers': 6, 'n_heads': 8, 'd_ff': 1024, 
                     'dropout': 0.1, 'vocab_size': 256},
            'compressive_memory': {'window_size': 512, 'compression_rate': 4}
        }
    
    codec = VortexCodec(
        **config['model'],
        **config['compressive_memory']
    ).to(device)
    
    if 'model_state_dict' in checkpoint:
        codec.load_state_dict(checkpoint['model_state_dict'])
        if 'epoch' in checkpoint:
            print(f"  Loaded from epoch {checkpoint['epoch']}")
        if 'val_bpd' in checkpoint:
            print(f"  Validation BPD: {checkpoint['val_bpd']:.4f}")
    else:
        codec.load_state_dict(checkpoint)
    
    print()
    
    try:
        stats = codec.compress_file(
            input_path=str(input_path),
            output_path=args.output,
            chunk_size=args.chunk_size,
            max_bytes=args.max_bytes
        )
        
        print()
        print("=" * 70)
        print("Compression Complete")
        print("=" * 70)
        print(f"Original size:    {stats['original_size']:,} bytes")
        print(f"Compressed size:  {stats['compressed_size']:,} bytes")
        print(f"Compression ratio: {stats['ratio']:.4f} ({stats['factor']:.2f}x)")
        print(f"Space savings:    {stats['savings_pct']:.2f}%")
        print(f"Bits per byte:    {stats['bpd']:.4f}")
        print(f"Time:             {stats['compress_time']:.2f} seconds")
        print(f"Throughput:       {stats['throughput_mbps']:.2f} MB/s")
        print()
        print(f"Compressed file saved to: {args.output}")
        
    except Exception as e:
        print(f"\nError during compression: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
