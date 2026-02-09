
"""Quick demo of compression."""

import torch
import numpy as np
from pathlib import Path
import time

from vortex.core import VortexCodec
from vortex.utils.metrics import evaluate_compression_performance


def create_sample_data(size_bytes=10000):
    """Make some fake data with patterns.\"\"\"
    base_pattern = np.array([i % 256 for i in range(100)], dtype=np.uint8)
    
    noise = np.random.randint(0, 50, size=100, dtype=np.int32)
    pattern = ((base_pattern.astype(np.int32) + noise) % 256).astype(np.uint8)
    
    num_repeats = size_bytes // len(pattern)
    data = np.tile(pattern, num_repeats)
    
    remaining = size_bytes - len(data)
    if remaining > 0:
        data = np.concatenate([data, pattern[:remaining]])
    
    return data


def demo_compression():
    print("=" * 70)
    print("Vortex-Codec Compression Demo")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    print("Initializing Vortex-Codec model...")
    codec = VortexCodec(
        d_model=256,
        n_layers=6,
        n_heads=8,
        d_ff=1024,
        window_size=512,
        compression_rate=4
    ).to(device)
    
    num_params = sum(p.numel() for p in codec.parameters())
    print(f"Model initialized with {num_params:,} parameters\n")
    
    print("Generating sample structured binary data...")
    sample_data = create_sample_data(size_bytes=5000)
    print(f"Created {len(sample_data)} bytes of synthetic telemetry data\n")
    
    print("Note: For actual compression, you need a trained model.")
    print("This demo shows the architecture in action with random weights.\n")
    
    codec.eval()
    with torch.no_grad():
        data_tensor = torch.from_numpy(sample_data).long().unsqueeze(0).to(device)
        
        print("Running forward pass to predict byte distributions...")
        start_time = time.time()
        logits, memories = codec(data_tensor)
        inference_time = time.time() - start_time
        
        print(f"Forward pass completed in {inference_time:.3f} seconds")
        print(f"Output shape: {logits.shape}")
        print(f"Memory buffers: {len(memories)} layers")
        
        if memories[0] is not None:
            print(f"Compressed memory shape: {memories[0].shape}\n")
        
        from vortex.utils.metrics import compute_bpd
        bpd = compute_bpd(logits[:, :-1], data_tensor[:, 1:])
        print(f"Bits-per-dimension (untrained): {bpd:.3f} bits/byte")
        print(f"Theoretical compression ratio: {bpd/8:.3f}\n")
    
    print("=" * 70)
    print("To use for actual compression:")
    print("  1. Train the model on your data: python train_example.py")
    print("  2. Load trained weights: codec.load('checkpoints/model.pt')")
    print("  3. Compress: compressed = codec.compress(data)")
    print("=" * 70)


def demo_dataset():
    print("\n" + "=" * 70)
    print("ByteDataset Demo")
    print("=" * 70)
    
    sample_file = Path("sample_telemetry.bin")
    
    if not sample_file.exists():
        print("Creating sample binary file...")
        sample_data = create_sample_data(size_bytes=50000)
        with open(sample_file, "wb") as f:
            f.write(sample_data.tobytes())
        print(f"Created {sample_file} ({sample_data.nbytes} bytes)\n")
    
    from vortex.io import ByteDataset
    
    print("Loading data with ByteDataset (window_size=512, stride=256)...")
    dataset = ByteDataset(
        file_path=sample_file,
        window_size=512,
        stride=256
    )
    
    print(f"Dataset contains {len(dataset)} windows")
    print(f"Total bytes: {dataset.total_bytes}")
    print(f"Vocabulary size: {dataset.vocab_size}\n")
    
    sample_window = dataset[0]
    print(f"Sample window shape: {sample_window.shape}")
    print(f"First 20 bytes: {sample_window[:20].tolist()}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    demo_compression()
    demo_dataset()
