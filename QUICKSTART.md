# Vortex-Codec: Quick Start Guide

## Complete End-to-End Workflow

### 1. Generate Test Datasets

Create realistic binary datasets including ATLAS-style detector events:

```bash
python generate_test_data.py
```

This generates:
- `data/atlas_events.bin` - High-energy detector event data (5.6 MB)
- `data/sensor_telemetry.bin` - Time-series sensor readings (35.7 MB) 
- `data/network_capture.bin` - Network packet captures (14.2 MB)
- `data/system_logs.bin` - Binary structured logs (2.1 MB)

### 2. Train a Model

Train Vortex-Codec on ATLAS data:

```bash
python train_example.py --data data/atlas_events.bin --epochs 10 --batch-size 16
```

Train on sensor telemetry:

```bash
python train_example.py --data data/sensor_telemetry.bin --epochs 10 --batch-size 16 --name sensors
```

Training options:
- `--epochs N` - Number of training epochs
- `--batch-size N` - Batch size (default: 32)
- `--lr FLOAT` - Learning rate (default: 0.0003)
- `--name NAME` - Experiment name for checkpoints
- `--resume PATH` - Resume from checkpoint
- `--eval-split FLOAT` - Validation split (default: 0.1)

Checkpoints save to: `checkpoints/<experiment_name>/`

### 3. Evaluate Compression Performance

#### Baseline Comparison (No Model Required)

Compare Gzip and Zstd on your data:

```bash
python evaluate.py --data data/atlas_events.bin --max-bytes 10000000
```

#### With Trained Model

Evaluate neural compression model:

```bash
python evaluate.py \
  --data data/atlas_events.bin \
  --model checkpoints/atlas_events/best_model.pt \
  --output results/atlas_results.json
```

#### Benchmark All Datasets

Run benchmarks on all generated datasets:

```bash
python benchmark_all.py
```

Results save to `results/` directory.

### 4. Example Training Session

```bash
# Generate test data
python generate_test_data.py

# Train on ATLAS events for 20 epochs
python train_example.py \
  --data data/atlas_events.bin \
  --epochs 20 \
  --batch-size 16 \
  --name atlas_20epochs

# Evaluate the trained model
python evaluate.py \
  --data data/atlas_events.bin \
  --model checkpoints/atlas_20epochs/best_model.pt \
  --max-bytes 5000000
```

## Understanding the Results

### Bits Per Dimension (BPD)

Lower is better. Theoretical minimum is the entropy of the data:
- **BPD < 2.0**: Excellent compression (4x+ improvement)
- **BPD 2.0-4.0**: Good compression (2-4x improvement)
- **BPD 4.0-6.0**: Moderate compression (1.3-2x improvement)
- **BPD > 6.0**: Minimal compression (<1.3x)

### Compression Ratio

Ratio of compressed to original size:
- **0.25**: 4:1 compression (75% savings)
- **0.50**: 2:1 compression (50% savings)
- **0.75**: 1.33:1 compression (25% savings)

### Baseline Performance (ATLAS Data)

From our test run on 1 MB of ATLAS events:
- **Gzip**: 7.32 BPD, 0.91 ratio (9% savings)
- **Zstd**: 7.29 BPD, 0.91 ratio (9% savings)

The ATLAS data is highly structured but has significant entropy. Standard codecs achieve minimal compression. This is where neural compression excels - by learning domain-specific patterns, Vortex-Codec can potentially achieve 2-4x better compression.

## Training Tips

### For Structured Binary Data (ATLAS-like)

```bash
python train_example.py \
  --data data/atlas_events.bin \
  --epochs 30 \
  --batch-size 8 \
  --lr 0.0001
```

Smaller batch size works better for structured event data with varying event sizes.

### For Time-Series Data (Sensor Telemetry)

```bash
python train_example.py \
  --data data/sensor_telemetry.bin \
  --epochs 20 \
  --batch-size 32 \
  --lr 0.0003
```

Larger batch size and higher learning rate work well for continuous time series.

### For Network Traffic

```bash
python train_example.py \
  --data data/network_capture.bin \
  --epochs 15 \
  --batch-size 16
```

Moderate settings work well for packet capture data.

## Expected Training Time

On NVIDIA GPU (RTX 3080 / A100):
- **ATLAS events (5.6 MB)**: ~5 minutes per epoch
- **Sensor telemetry (35.7 MB)**: ~30 minutes per epoch  
- **Network capture (14.2 MB)**: ~12 minutes per epoch

On CPU:
- Expect 10-20x slower

## Monitoring Training

The training script shows:
- **Loss**: Cross-entropy loss (lower is better)
- **BPD**: Bits per dimension (compression metric)
- **Progress bar**: Per-epoch progress with live metrics

Example output:
```
Epoch 5/20
Training: 100%|████████| 2573/2573 [04:32<00:00, 9.44it/s, loss=3.8234, bpd=5.5123]
  Train Loss: 3.8234 | Train BPD: 5.5123
  Val Loss:   3.7891 | Val BPD:   5.4658
  Time: 272.3s
  ✓ New best validation BPD: 5.4658
```

## Troubleshooting

### Out of Memory

Reduce batch size:
```bash
python train_example.py --data data/atlas_events.bin --batch-size 4
```

### Training Too Slow

Use smaller dataset chunk:
```bash
# Modify ByteDataset call to limit data size
# In train_example.py line ~140, the dataset is created
# You can add max_bytes parameter to limit data
```

### Model Not Learning (BPD Not Decreasing)

Try:
1. Lower learning rate: `--lr 0.0001`
2. Increase model capacity: Edit `configs/default.yaml` (increase `d_model` or `n_layers`)
3. Train for more epochs: `--epochs 50`

## Next Steps

1. **Optimize hyperparameters**: Edit `configs/default.yaml`
2. **Implement actual compression**: Currently evaluates BPD only. Full arithmetic coding integration in `vortex/core/codec.py`
3. **Add CLI tools**: Implement `bin/compress.py` and `bin/ decompress.py`
4. **Benchmark against Brotli**: Add Brotli support to `evaluate.py`
5. **Distributed training**: Train on larger datasets with multi-GPU

## Project Structure

```
vortex-codec/
├── generate_test_data.py   # Create test datasets
├── train_example.py         # Training script
├── evaluate.py              # Evaluation and benchmarking
├── benchmark_all.py         # Batch benchmark runner
├── demo.py                  # Interactive demo
├── data/                    # Generated datasets
│   ├── atlas_events.bin
│   ├── sensor_telemetry.bin
│   └── manifest.json
├── checkpoints/             # Saved models
│   └── <experiment_name>/
│       ├── best_model.pt
│       └── checkpoint_epoch_*.pt
└── results/                 # Benchmark results
    └── *_results.json
```
