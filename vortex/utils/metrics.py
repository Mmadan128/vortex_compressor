"""
Metrics and evaluation utilities for compression performance.

This module provides standard metrics for evaluating compression quality
including bits-per-dimension (BPD), compression ratio, and throughput
measurements.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Union, Tuple


def compute_bpd(
    model_output: torch.Tensor,
    target: torch.Tensor,
    reduction: str = 'mean'
) -> float:
    """
    Computes bits-per-dimension (BPD) metric for compression quality.
    
    BPD measures the average number of bits needed to encode each byte
    using the model's predicted probability distribution. Lower values
    indicate better compression potential.
    
    Args:
        model_output: Model logits [batch, seq_len, vocab_size]
        target: Target byte values [batch, seq_len]
        reduction: How to aggregate ('mean', 'sum', or 'none')
        
    Returns:
        BPD value (lower is better)
        
    Example:
        >>> logits = model(input_bytes)
        >>> bpd = compute_bpd(logits, target_bytes)
        >>> print(f"Model achieves {bpd:.3f} bits per byte")
    """
    batch_size, seq_len, vocab_size = model_output.shape
    
    log_probs = F.log_softmax(model_output, dim=-1)
    
    target_log_probs = log_probs.gather(
        dim=-1,
        index=target.unsqueeze(-1)
    ).squeeze(-1)
    
    nll = -target_log_probs
    
    bpd = nll / np.log(2)
    
    if reduction == 'mean':
        return bpd.mean().item()
    elif reduction == 'sum':
        return bpd.sum().item()
    else:
        return bpd


def compression_ratio(
    original_size: int,
    compressed_size: int
) -> float:
    """
    Computes compression ratio as percentage of original size.
    
    Args:
        original_size: Size of uncompressed data in bytes
        compressed_size: Size of compressed data in bytes
        
    Returns:
        Compression ratio (e.g., 0.25 means 4:1 compression)
        
    Example:
        >>> ratio = compression_ratio(original_size=10000, compressed_size=2500)
        >>> print(f"Achieved {ratio:.2%} compression (4x reduction)")
    """
    if original_size == 0:
        return 0.0
    
    return compressed_size / original_size


def compression_factor(
    original_size: int,
    compressed_size: int
) -> float:
    """
    Computes compression factor (inverse of ratio).
    
    Args:
        original_size: Size of uncompressed data in bytes
        compressed_size: Size of compressed data in bytes
        
    Returns:
        Compression factor (e.g., 4.0 means 4x size reduction)
    """
    if compressed_size == 0:
        return float('inf')
    
    return original_size / compressed_size


def percent_savings(
    original_size: int,
    compressed_size: int
) -> float:
    """
    Computes storage savings as percentage.
    
    Args:
        original_size: Size of uncompressed data in bytes
        compressed_size: Size of compressed data in bytes
        
    Returns:
        Percentage savings (e.g., 75.0 means 75% reduction in size)
    """
    if original_size == 0:
        return 0.0
    
    savings = (original_size - compressed_size) / original_size * 100
    return savings


def cross_entropy_loss(
    model_output: torch.Tensor,
    target: torch.Tensor,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Computes cross-entropy loss for next-byte prediction.
    
    Args:
        model_output: Model logits [batch, seq_len, vocab_size]
        target: Target byte values [batch, seq_len]
        reduction: Loss reduction method
        
    Returns:
        Cross-entropy loss tensor
    """
    batch_size, seq_len, vocab_size = model_output.shape
    
    model_output_flat = model_output.reshape(-1, vocab_size)
    target_flat = target.reshape(-1)
    
    loss = F.cross_entropy(
        model_output_flat,
        target_flat,
        reduction=reduction
    )
    
    return loss


def evaluate_compression_performance(
    original_data: Union[bytes, np.ndarray, torch.Tensor],
    compressed_data: bytes,
    bpd: float
) -> dict:
    """
    Computes comprehensive compression performance metrics.
    
    Args:
        original_data: Uncompressed byte data
        compressed_data: Compressed byte string
        bpd: Bits-per-dimension from model evaluation
        
    Returns:
        Dictionary containing multiple metrics:
        - original_size: Size in bytes
        - compressed_size: Size in bytes
        - compression_ratio: Fraction of original
        - compression_factor: Inverse of ratio
        - percent_savings: Percentage reduction
        - bpd: Bits per byte
        - effective_bpd: Actual bits per byte achieved
    """
    if isinstance(original_data, bytes):
        original_size = len(original_data)
    elif isinstance(original_data, np.ndarray):
        original_size = original_data.size
    elif isinstance(original_data, torch.Tensor):
        original_size = original_data.numel()
    else:
        raise TypeError("Unsupported data type for original_data")
    
    compressed_size = len(compressed_data)
    
    ratio = compression_ratio(original_size, compressed_size)
    factor = compression_factor(original_size, compressed_size)
    savings = percent_savings(original_size, compressed_size)
    effective_bpd = (compressed_size * 8) / original_size
    
    return {
        'original_size_bytes': original_size,
        'compressed_size_bytes': compressed_size,
        'compression_ratio': ratio,
        'compression_factor': factor,
        'percent_savings': savings,
        'model_bpd': bpd,
        'effective_bpd': effective_bpd
    }
