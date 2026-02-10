# Vortex-Codec

Neural lossless compressor for binary data. Uses transformer + arithmetic coding to beat gzip/zstd on structured datasets.

## What it does

Learns byte patterns in your data and compresses better than standard codecs. Especially good for repetitive binary formats like sensor logs, telemetry, network captures.



## How it works

1. Transformer predicts next byte probabilities
2. Arithmetic coder compresses using those predictions
3. Compressive attention keeps memory usage reasonable

## Install

```bash
git clone https://github.com/yourusername/vortex-codec.git
cd vortex-codec

pip install torch torchvision torchaudio torchac pyyaml numpy tqdm
```

## Usage

### Train a model

```bash
python train_example.py --data mydata.bin --epochs 10 --batch-size 16
```

### Compress a file

```bash
python -m bin.compress input.bin -o output.vortex -m checkpoints/best_model.pt
```

```python
from vortex.io import ByteDataset
from vortex.core import VortexCodec
from vortex.utils.metrics import compute_bpd
from torch.utils.data import DataLoader
import yaml

with open("configs/default.yaml") as f:
    config = yaml.safe_load(f)

dataset = ByteDataset(
    file_path="industrial_logs.bin",
    window_size=config['dataset']['window_size'],
    stride=config['dataset']['stride']
)

loader = DataLoader(dataset, batch_size=32, shuffle=True)

codec = VortexCodec(**config['model'], **config['compressive_memory'])
codec.train()

optimizer = torch.optim.AdamW(codec.parameters(), lr=3e-4)

# Initialize compressed memories (critical for compressive transformers!)
compressed_memories = None

for batch in loader:
    # Maintain memory state across batches - this is the key to compressive transformers
    logits, compressed_memories = codec(batch, compressed_memories=compressed_memories)
    
    # Detach memories to prevent backprop through entire history
    compressed_memories = [m.detach() if m is not None else None 
                          for m in compressed_memories]
    
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, 256),
        batch[:, 1:].reshape(-1)
    )
    
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    bpd = compute_bpd(logits[:, :-1], batch[:, 1:])
    print(f"BPD: {bpd:.3f}")

## Config

Edit `configs/default.yaml`:

```yaml
model:
  d_model: 256      # model size
  n_layers: 6       # transformer layers
  n_heads: 8        # attention heads

compressive_memory:
  window_size: 512      # context window
  compression_rate: 4   # memory compression (4:1)
```

## Project Structure

```
vortex-codec/
├── bin/              # compress/decompress tools
├── vortex/
│   ├── modules/      # attention + memory
│   ├── core/         # main codec
│   ├── io/           # data loading
│   └── utils/        # metrics
├── configs/
├── experiments/      # ATLAS data
└── train_example.py  # training script
```

## How it works

1. **Transformer** predicts next byte probabilities
2. **Arithmetic coder** uses those probabilities to compress
3. **Compressive attention** keeps old context without eating all your RAM

The model learns patterns specific to your data type, so it compresses better than generic codecs.

## TODO

- [x] Compressive transformer
- [x] Arithmetic coding with torchac
- [x] Training pipeline
- [ ] Decompression (currently compression-only)
- [ ] Faster inference
- [ ] More benchmarks

## License

MIT
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/vortex-codec}
}
```

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please open an issue before submitting major changes.

## Acknowledgments

- Compressive Transformer architecture inspired by foundational work in efficient attention mechanisms
- `torchac` library for high-performance arithmetic coding
- PyTorch team for the underlying deep learning framework
