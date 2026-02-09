# Benchmark Scripts

This directory will contain scripts for benchmarking Vortex-Codec against
standard compression algorithms:

- `benchmark_gzip.py` - Compare against Gzip
- `benchmark_zstd.py` - Compare against Zstandard
- `benchmark_brotli.py` - Compare against Brotli
- `ablation_studies.py` - Test architectural variations

## Running Benchmarks

```bash
python experiments/benchmark_gzip.py --data industrial_logs.bin --model weights.pt
```

Results will include:
- Compression ratio
- Compression/decompression throughput (MB/s)
- Bits-per-dimension (BPD)
- Peak memory usage
