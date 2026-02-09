"""
Compressive Transformer components for long-sequence modeling.

This module implements the core compressive attention mechanism that maintains
both recent activations and compressed long-term memory. When the local attention
window fills, older activations are compressed and moved to a separate memory buffer,
enabling the model to capture patterns across very long byte sequences without
quadratic memory growth.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class MemoryManager(nn.Module):
    """
    Manages compression and storage of long-term attention activations.
    
    When the recent activation window reaches capacity, this module compresses
    the oldest activations using learnable strided convolution and stores them
    in a secondary memory buffer. This compression preserves long-range pattern
    information while reducing memory requirements.
    
    Args:
        d_model: Dimension of transformer hidden states
        compression_rate: Ratio of length reduction (e.g., 4 means 4:1 compression)
        max_compressed_len: Maximum length of compressed memory buffer
    """
    
    def __init__(
        self,
        d_model: int,
        compression_rate: int = 4,
        max_compressed_len: int = 512
    ):
        super().__init__()
        self.d_model = d_model
        self.compression_rate = compression_rate
        self.max_compressed_len = max_compressed_len
        
        self.compression_layer = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=compression_rate,
            stride=compression_rate,
            padding=0
        )
        
        self.layer_norm = nn.LayerNorm(d_model)
    
    def compress_activations(self, activations: torch.Tensor) -> torch.Tensor:
        """
        Compresses a sequence of activations using strided convolution.
        
        Args:
            activations: Tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            Compressed tensor of shape [batch_size, seq_len // compression_rate, d_model]
        """
        batch_size, seq_len, d_model = activations.shape
        
        activations_transposed = activations.transpose(1, 2)
        compressed = self.compression_layer(activations_transposed)
        compressed = compressed.transpose(1, 2)
        compressed = self.layer_norm(compressed)
        
        return compressed
    
    def update_memory(
        self,
        new_activations: torch.Tensor,
        old_memory: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates memory buffer by compressing new activations and managing buffer size.
        
        Args:
            new_activations: Recent activations to compress [batch, seq_len, d_model]
            old_memory: Existing compressed memory buffer or None
            
        Returns:
            Tuple of (compressed_new, updated_memory)
        """
        compressed_new = self.compress_activations(new_activations)
        
        if old_memory is None:
            return compressed_new, compressed_new
        
        updated_memory = torch.cat([old_memory, compressed_new], dim=1)
        
        if updated_memory.size(1) > self.max_compressed_len:
            updated_memory = updated_memory[:, -self.max_compressed_len:, :]
        
        return compressed_new, updated_memory


class CompressiveAttention(nn.Module):
    """
    Multi-head attention layer with compressive memory for long sequences.
    
    This layer maintains two distinct memory banks:
    1. Recent activations: Full-resolution attention over recent context
    2. Compressed memory: Down-sampled activations from distant past
    
    The model attends to both banks, allowing it to capture both fine-grained
    local patterns and coarse long-term trends in the byte stream.
    
    Args:
        d_model: Dimension of model hidden states
        n_heads: Number of attention heads
        window_size: Size of recent activation window
        compression_rate: Compression ratio for old activations
        dropout: Dropout probability for attention weights
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        window_size: int = 512,
        compression_rate: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size
        self.compression_rate = compression_rate
        self.scale = math.sqrt(self.head_dim)
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        self.memory_manager = MemoryManager(
            d_model=d_model,
            compression_rate=compression_rate,
            max_compressed_len=window_size
        )
    
    def forward(
        self,
        x: torch.Tensor,
        compressed_memory: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with compressive attention over recent and compressed activations.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            compressed_memory: Optional compressed history [batch, mem_len, d_model]
            mask: Optional attention mask [batch, seq_len, total_len]
            
        Returns:
            Tuple of (output, updated_compressed_memory)
        """
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x)
        
        if compressed_memory is not None:
            kv_input = torch.cat([compressed_memory, x], dim=1)
        else:
            kv_input = x
        
        k = self.k_proj(kv_input)
        v = self.v_proj(kv_input)
        
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        
        output = self.out_proj(attn_output)
        
        return output, compressed_memory
    
    def compress_and_update_memory(
        self,
        activations: torch.Tensor,
        old_memory: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compresses activations and updates the memory buffer.
        
        Args:
            activations: Activations to compress [batch, seq_len, d_model]
            old_memory: Existing memory buffer or None
            
        Returns:
            Updated compressed memory buffer
        """
        _, updated_memory = self.memory_manager.update_memory(activations, old_memory)
        return updated_memory


class CompressiveTransformerBlock(nn.Module):
    """
    Single transformer block with compressive attention and feed-forward network.
    
    Implements the standard transformer architecture with pre-layer normalization
    and residual connections, but uses CompressiveAttention instead of standard
    multi-head attention.
    
    Args:
        d_model: Model hidden dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward network hidden dimension
        window_size: Recent activation window size
        compression_rate: Memory compression ratio
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_ff: int = 2048,
        window_size: int = 512,
        compression_rate: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.attention = CompressiveAttention(
            d_model=d_model,
            n_heads=n_heads,
            window_size=window_size,
            compression_rate=compression_rate,
            dropout=dropout
        )
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        compressed_memory: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through transformer block with compressive attention.
        
        Args:
            x: Input tensor [batch, seq_len, d_model]
            compressed_memory: Optional compressed memory
            mask: Optional attention mask
            
        Returns:
            Tuple of (output, updated_memory)
        """
        normed_x = self.norm1(x)
        attn_output, updated_memory = self.attention(
            normed_x,
            compressed_memory=compressed_memory,
            mask=mask
        )
        x = x + self.dropout(attn_output)
        
        normed_x = self.norm2(x)
        ff_output = self.feed_forward(normed_x)
        x = x + ff_output
        
        return x, updated_memory
