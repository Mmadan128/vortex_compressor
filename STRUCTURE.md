# Vortex-Codec Project Structure

## Complete Directory Tree

```
vortex-codec/
│
├── README.md                    # Main project documentation
├── LICENSE                      # MIT License
├── requirements.txt             # Python dependencies
├── setup.py                     # Package installation configuration
├── .gitignore                   # Git ignore rules
│
├── train_example.py             # Example training script
├── demo.py                      # Interactive demonstration script
│
├── vortex/                      # Core package
│   ├── __init__.py
│   │
│   ├── modules/                 # Neural network components
│   │   ├── __init__.py
│   │   └── compressive.py       # CompressiveAttention, MemoryManager, TransformerBlock
│   │
│   ├── core/                    # Compression engine
│   │   ├── __init__.py
│   │   └── codec.py             # VortexCodec, PositionalEncoding, torchac integration
│   │
│   ├── io/                      # Data loading
│   │   ├── __init__.py
│   │   └── dataset.py           # ByteDataset, StreamingByteDataset
│   │
│   └── utils/                   # Utilities
│       ├── __init__.py
│       └── metrics.py           # BPD, compression ratio, evaluation functions
│
├── configs/                     # Configuration files
│   ├── default.yaml             # Default training/model config
│   └── ablation.yaml            # Ablation study configurations
│
├── bin/                         # Command-line tools
│   ├── __init__.py
│   ├── compress.py              # CLI compressor (stub)
│   └── decompress.py            # CLI decompressor (stub)
│
└── experiments/                 # Benchmarking scripts
    └── README.md                # Benchmark documentation
```

## Core Components

### 1. Neural Network Modules (`vortex/modules/`)

- **MemoryManager**: Compresses old activations using strided Conv1d
- **CompressiveAttention**: Multi-head attention with dual memory banks
- **CompressiveTransformerBlock**: Full transformer layer with FFN

### 2. Compression Engine (`vortex/core/`)

- **VortexCodec**: High-level interface for compress/decompress
- **PositionalEncoding**: Sinusoidal position embeddings
- Integration with `torchac` for arithmetic coding

### 3. Data I/O (`vortex/io/`)

- **ByteDataset**: In-memory sliding-window dataset
- **StreamingByteDataset**: Memory-mapped dataset for large files

### 4. Metrics (`vortex/utils/`)

- Bits-per-dimension (BPD)
- Compression ratio, factor, and savings
- Cross-entropy loss
- Comprehensive performance evaluation

## Key Files

### Configuration (`configs/default.yaml`)

Defines all hyperparameters:
- Model architecture (d_model, n_layers, n_heads)
- Compressive memory settings (window_size, compression_rate)
- Training parameters (learning rate, batch size)

### Training Script (`train_example.py`)

Complete example showing:
- Dataset loading
- Model initialization
- Training loop with gradient clipping
- Checkpoint saving

### Demo Script (`demo.py`)

Interactive demo that:
- Creates synthetic telemetry data
- Shows forward pass and memory management
- Demonstrates dataset usage

## Installation

```bash
pip install -r requirements.txt
```

Or for development:

```bash
pip install -e .
```

## Usage

### Python API

```python
from vortex.core import VortexCodec

codec = VortexCodec(d_model=256, n_layers=6)
codec.load("weights.pt")

compressed = codec.compress(raw_bytes)
decompressed = codec.decompress(compressed, target_length=len(raw_bytes))
```

### Training

```bash
python train_example.py
```

### Demo

```bash
python demo.py
```

## Dependencies

- PyTorch >= 2.0.0
- torchac >= 0.9.3 (arithmetic coding)
- NumPy >= 1.21.0
- PyYAML >= 6.0

## Architecture Highlights

### Compressive Memory Mechanism

```
Input Bytes → Embedding → Positional Encoding
                                ↓
    ┌─────────────────────────────────────────┐
    │    Compressive Transformer Blocks       │
    │                                          │
    │  Recent Memory ── Compress ──► Long-Term │
    │   [512 tokens]    (4:1)      [512 tokens]│
    └─────────────────────────────────────────┘
                                ↓
               Output Projection (256-way softmax)
                                ↓
                 CDF → Arithmetic Coding → Compressed Bytes
```

### Memory Compression

When recent window fills (512 tokens):
1. Oldest 128 tokens extracted
2. Compressed via strided Conv1d (4→1)
3. Result (32 tokens) added to long-term memory
4. Long-term memory capped at 512 tokens

This maintains $O(n)$ memory while preserving distant patterns.

## Future Extensions

- [ ] CLI tools implementation
- [ ] Distributed training support
- [ ] Benchmark suite (Gzip/Zstd/Brotli)
- [ ] Model quantization
- [ ] Online learning mode
- [ ] Streaming compression API
