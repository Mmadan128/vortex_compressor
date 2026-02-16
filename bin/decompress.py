"""
CLI entry point for Vortex decompression.

Usage:
    python -m bin.decompress input.vortex -o output.dat --model weights.pt
"""

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
        description="Decompress Vortex-encoded data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Decompress with trained model
  python -m bin.decompress data.vortex -o data.bin -m model.pt
  
  # Decompress with specific config
  python -m bin.decompress compressed.vortex -o output.bin -m model.pt --config config.yaml

Note: The model used for decompression must be the same as the one used for
compression, with matching architecture and weights.
"""
    )
    
    parser.add_argument(
        "input",
        type=str,
        help="Path to compressed .vortex file"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Path to output binary file"
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        required=True,
        help="Path to trained model weights (.pt file)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (optional if checkpoint has config)"
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
    print("Vortex-Codec Decompression")
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
        stats = codec.decompress_file(
            input_path=str(input_path),
            output_path=args.output
        )
        
        print()
        print("=" * 70)
        print("Decompression Complete")
        print("=" * 70)
        print(f"Compressed size:    {stats['compressed_size']:,} bytes")
        print(f"Decompressed size:  {stats['decompressed_size']:,} bytes")
        print(f"Compression ratio:  {stats['ratio']:.4f} ({stats['factor']:.2f}x)")
        print(f"Time:               {stats['decompress_time']:.2f} seconds")
        print(f"Throughput:         {stats['throughput_mbps']:.2f} MB/s")
        print()
        print(f"Decompressed file saved to: {args.output}")
        
    except Exception as e:
        print(f"\nError during decompression: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
