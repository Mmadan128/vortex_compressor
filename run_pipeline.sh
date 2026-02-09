#!/bin/bash

# Vortex-Codec: Full Pipeline Test
# This script runs the complete pipeline from data generation to evaluation

set -e

echo "======================================================================="
echo "Vortex-Codec Complete Pipeline Test"
echo "======================================================================="

echo ""
echo "[1/4] Generating test datasets..."
python generate_test_data.py

echo ""
echo "[2/4] Running baseline benchmarks on ATLAS data..."
python evaluate.py --data data/atlas_events.bin --max-bytes 1000000

echo ""
echo "[3/4] Training model on ATLAS data (2 epochs for quick test)..."
python train_example.py \
  --data data/atlas_events.bin \
  --epochs 2 \
  --batch-size 8 \
  --name quick_test

echo ""
echo "[4/4] Evaluating trained model..."
python evaluate.py \
  --data data/atlas_events.bin \
  --model checkpoints/quick_test/best_model.pt \
  --max-bytes 1000000 \
  --output results/quick_test_results.json

echo ""
echo "======================================================================="
echo "Pipeline Test Complete!"
echo "======================================================================="
echo ""
echo "Generated datasets:"
ls -lh data/*.bin
echo ""
echo "Model checkpoints:"
ls -lh checkpoints/quick_test/
echo ""
echo "Results:"
cat results/quick_test_results.json
echo ""
echo "To train for longer:"
echo "  python train_example.py --data data/atlas_events.bin --epochs 20"
