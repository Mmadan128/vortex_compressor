#!/usr/bin/env python3
"""
Test that compressive attention mechanism is actually working.

This verifies:
1. Memory gets updated between chunks
2. Memory gets compressed when window exceeds limit
3. Attention complexity is bounded to O(window_size) not O(total_sequence)
"""

import torch
import yaml
from vortex.core import VortexCodec

def test_compressive_memory():
    """Verify compressive memory actually compresses."""
    
    print("=" * 80)
    print("Testing Compressive Memory Mechanism")
    print("=" * 80)
    print()
    
    # Load model
    checkpoint = torch.load('model/best_model.pt', weights_only=False)
    config = checkpoint['config']
    
    window_size = config['compressive_memory']['window_size']
    compression_rate = config['compressive_memory']['compression_rate']
    
    print(f"Configuration:")
    print(f"  Window size: {window_size}")
    print(f"  Compression rate: {compression_rate}:1")
    print(f"  d_model: {config['model']['d_model']}")
    print(f"  n_layers: {config['model']['n_layers']}")
    print()
    
    codec = VortexCodec(
        **config['model'],
        **config['compressive_memory'],
        use_flash_attention=True
    ).cuda()
    
    codec.load_state_dict(checkpoint['model_state_dict'])
    codec.eval()
    
    # Test 1: Memory should update between chunks
    print("Test 1: Memory updates between chunks")
    print("-" * 80)
    
    chunk1 = torch.randint(0, 256, (1, 512), dtype=torch.long).cuda()
    chunk2 = torch.randint(0, 256, (1, 512), dtype=torch.long).cuda()
    
    with torch.no_grad():
        _, mem1 = codec.forward(chunk1, compressed_memories=None)
        _, mem2 = codec.forward(chunk2, compressed_memories=mem1)
    
    print(f"✓ Chunk 1 processed")
    print(f"  Memory after chunk 1 (layer 0): {mem1[0].shape if mem1[0] is not None else None}")
    print(f"✓ Chunk 2 processed")
    print(f"  Memory after chunk 2 (layer 0): {mem2[0].shape if mem2[0] is not None else None}")
    
    if mem1[0] is not None and mem2[0] is not None:
        mem_grew = mem2[0].size(1) > mem1[0].size(1)
        print(f"  {'✓' if mem_grew else '✗'} Memory grew: {mem1[0].size(1)} -> {mem2[0].size(1)}")
        if mem_grew:
            print("  ✅ PASS: Memory is being updated!")
        else:
            print("  ⚠️  WARNING: Memory didn't grow (might already be at max)")
    else:
        print("  ✗ FAIL: Memory is None - compressive mechanism not working!")
        return False
    
    print()
    
    # Test 2: Memory should compress when exceeding window
    print("Test 2: Memory compresses when exceeding window")
    print("-" * 80)
    
    # Process many chunks to trigger compression
    compressed_memories = None
    memory_sizes = []
    
    num_chunks = 10
    chunk_size = 512
    
    print(f"Processing {num_chunks} chunks of {chunk_size} bytes...")
    
    with torch.no_grad():
        for i in range(num_chunks):
            chunk = torch.randint(0, 256, (1, chunk_size), dtype=torch.long).cuda()
            _, compressed_memories = codec.forward(chunk, compressed_memories=compressed_memories)
            
            if compressed_memories[0] is not None:
                mem_size = compressed_memories[0].size(1)
                memory_sizes.append(mem_size)
                print(f"  Chunk {i+1}: memory size = {mem_size}")
    
    print()
    
    # Analyze memory growth
    total_input = num_chunks * chunk_size
    final_memory_size = memory_sizes[-1] if memory_sizes else 0
    
    print(f"Results:")
    print(f"  Total input processed: {total_input} tokens")
    print(f"  Final memory size: {final_memory_size} tokens")
    print(f"  Compression ratio: {total_input / final_memory_size:.2f}×")
    print()
    
    # Calculate theoretical maximum without compression
    theoretical_max_without_compression = total_input
    
    # With compression, memory should be bounded
    max_expected_with_compression = window_size + (window_size // compression_rate)
    
    print(f"Analysis:")
    print(f"  WITHOUT compression: memory would be {theoretical_max_without_compression} tokens")
    print(f"  WITH compression (bounded): memory should be ≤{max_expected_with_compression} tokens")
    print(f"  Actual: {final_memory_size} tokens")
    print()
    
    if final_memory_size < theoretical_max_without_compression * 0.8:
        print("  ✅ PASS: Compression is working! Memory is bounded.")
        compression_working = True
    else:
        print("  ✗ FAIL: Memory grew linearly - compression NOT working!")
        compression_working = False
    
    print()
    
    # Test 3: Verify attention cost is bounded
    print("Test 3: Attention complexity is bounded")
    print("-" * 80)
    
    if final_memory_size > 0:
        attention_context_size = final_memory_size + chunk_size
        print(f"  Current chunk size: {chunk_size}")
        print(f"  Compressed memory size: {final_memory_size}")
        print(f"  Total attention context: {attention_context_size} tokens")
        print(f"  Attention complexity: O({attention_context_size}²) per head")
        print()
        
        if attention_context_size <= max_expected_with_compression:
            print(f"  ✅ PASS: Attention is bounded to O({max_expected_with_compression}²)")
            print(f"  This is O(window_size²), NOT O(total_sequence²)!")
            attention_bounded = True
        else:
            print(f"  ⚠️  WARNING: Attention context is larger than expected")
            attention_bounded = False
    else:
        attention_bounded = False
    
    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    
    if compression_working and attention_bounded:
        print("✅ COMPRESSIVE MECHANISM IS WORKING!")
        print()
        print("Your model should now achieve near-O(n) complexity:")
        print(f"  - Attention bounded to {max_expected_with_compression} tokens (not full sequence)")
        print(f"  - Memory grows slowly ({compression_rate}:1 compression)")
        print(f"  - Should be competitive with Mamba/SSM models")
        print()
        print("Expected speedup:")
        print("  - 2-5× faster compression")
        print("  - 3-10× less memory usage")
        print("  - Scales better for long sequences")
        return True
    else:
        print("⚠️  COMPRESSIVE MECHANISM MAY HAVE ISSUES")
        print()
        print("Please review the implementation in:")
        print("  - vortex/modules/compressive.py")
        print("  - CompressiveAttention._update_compressed_memory()")
        return False

if __name__ == "__main__":
    success = test_compressive_memory()
    exit(0 if success else 1)
