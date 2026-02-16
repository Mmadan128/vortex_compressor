#!/bin/bash
# Find the fastest compression configuration
# Tests different chunk_size and batch_size combinations

set -e

INPUT="experiments/atlas_experiment/atlas_10m.bin"
MODEL="model/best_model.pt"
MAX_BYTES=5000000  # Test with 5MB for speed

echo "🔍 Finding fastest configuration for Vortex compression..."
echo "Test file: $INPUT (first ${MAX_BYTES} bytes)"
echo ""

# Test configurations (chunk_size, batch_size)
CONFIGS=(
    "2048 64"
    "4096 64"
    "8192 64"
    "16384 64"
    "8192 128"
    "16384 128"
    "16384 256"
)

BEST_TIME=999999
BEST_CONFIG=""

for config in "${CONFIGS[@]}"; do
    read chunk batch <<< "$config"
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Testing: chunk_size=$chunk, batch_size=$batch"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Run compression and capture time
    START=$(date +%s.%N)
    
    python bin/compress_optimized.py \
        "$INPUT" \
        -o "test_config_${chunk}_${batch}.vxc" \
        -m "$MODEL" \
        --max-bytes "$MAX_BYTES" \
        --chunk-size "$chunk" \
        --batch-size "$batch" \
        --compile \
        2>&1 | grep -E "(Total time|Throughput|Compression ratio)" || true
    
    END=$(date +%s.%N)
    ELAPSED=$(echo "$END - $START" | bc)
    
    echo "⏱  Wall time: ${ELAPSED}s"
    echo ""
    
    # Track best
    if (( $(echo "$ELAPSED < $BEST_TIME" | bc -l) )); then
        BEST_TIME=$ELAPSED
        BEST_CONFIG="chunk_size=$chunk batch_size=$batch"
    fi
    
    # Clean up test file
    rm -f "test_config_${chunk}_${batch}.vxc"
done

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🏆 RESULTS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Best configuration: $BEST_CONFIG"
echo "Best time: ${BEST_TIME}s"
echo ""
echo "Recommended command:"
echo "python bin/compress_optimized.py \\"
echo "    INPUT_FILE \\"
echo "    -o OUTPUT_FILE \\"
echo "    -m $MODEL \\"
grep -o "batch_size=[0-9]*" <<< "$BEST_CONFIG" | sed 's/batch_size=/    --batch-size /' | tr -d '\n'
echo " \\"
grep -o "chunk_size=[0-9]*" <<< "$BEST_CONFIG" | sed 's/chunk_size=/    --chunk-size /' | tr -d '\n'
echo " \\"
echo "    --compile"
echo ""
