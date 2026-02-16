"""
Compressive Transformer components for long-sequence modeling.

Based on "Compressive Transformers for Long-Range Sequence Modelling"
(Rae et al., 2019) - https://arxiv.org/abs/1911.05507

This module implements the core compressive attention mechanism that maintains
both recent activations and compressed long-term memory. When the local attention
window fills, older activations are compressed and moved to a separate memory buffer,
enabling the model to capture patterns across very long byte sequences without
quadratic memory growth.

Key implementation notes:
    - Memory state must persist across forward passes (batches/chunks)
    - Old activations are compressed using learnable Conv1D with stride
    - Compressed memory is concatenated with recent context for attention
    - Memory should be detached between batches to prevent backprop through entire history
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Literal
import math
import warnings


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
        use_flash_attention: Enable Flash Attention 3 for faster inference (requires PyTorch 2.0+)
        flash_backend: Flash attention backend ('flash', 'mem_efficient', 'math', or 'auto')
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        window_size: int = 512,
        compression_rate: int = 4,
        dropout: float = 0.1,
        use_flash_attention: bool = True,
        flash_backend: Literal['flash', 'mem_efficient', 'math', 'auto'] = 'auto'
    ):
        super().__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size
        self.compression_rate = compression_rate
        self.scale = math.sqrt(self.head_dim)
        self.dropout_p = dropout
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Flash Attention configuration
        self.use_flash_attention = use_flash_attention
        self.flash_backend = flash_backend
        
        # Check Flash Attention availability
        if self.use_flash_attention:
            try:
                # Test if scaled_dot_product_attention is available
                with torch.backends.cuda.sdp_kernel(
                    enable_flash=True,
                    enable_math=True,
                    enable_mem_efficient=True
                ):
                    pass
                self._flash_available = True
            except (AttributeError, RuntimeError):
                warnings.warn(
                    "Flash Attention requested but not available in this PyTorch version. "
                    "Falling back to standard attention. Consider upgrading to PyTorch 2.0+"
                )
                self._flash_available = False
                self.use_flash_attention = False
        else:
            self._flash_available = False
        
        # Memory manager for compressive attention
        # max_compressed_len should be larger than window_size to allow compressed history
        # Typical: 2-3× window_size for good long-range context with bounded attention
        self.memory_manager = MemoryManager(
            d_model=d_model,
            compression_rate=compression_rate,
            max_compressed_len=window_size * 2  # Allow compressed history to accumulate
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
        
        # Build key-value context: compressed memory + current input
        if compressed_memory is not None:
            kv_input = torch.cat([compressed_memory, x], dim=1)
        else:
            kv_input = x
        
        k = self.k_proj(kv_input)
        v = self.v_proj(kv_input)
        
        # Reshape for multi-head attention: [batch, n_heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Use Flash Attention if available and enabled
        if self.use_flash_attention and self._flash_available:
            attn_output = self._flash_attention(q, k, v, mask)
        else:
            attn_output = self._standard_attention(q, k, v, mask)
        
        # Reshape back: [batch, seq_len, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        
        output = self.out_proj(attn_output)
        
        # ===================================================================
        # CRITICAL FIX: Actually use the compressive memory mechanism!
        # ===================================================================
        # Update compressed memory by:
        # 1. Add new activations (x) to the memory pool
        # 2. If exceeds window_size, compress oldest activations
        # 3. Keep compressed memory bounded to max_compressed_len
        
        # Use the post-projection output as the "activations" to remember
        updated_memory = self._update_compressed_memory(output, compressed_memory)
        
        return output, updated_memory
    
    def _update_compressed_memory(
        self,
        new_activations: torch.Tensor,
        old_memory: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Update the compressed memory with new activations.
        
        This is the KEY to achieving O(n) complexity:
        - Maintain a window of recent activations (window_size)
        - When window fills, compress oldest activations (4:1 compression)
        - Compressed memory grows slowly, keeping attention cost bounded
        
        Args:
            new_activations: New activations to add [batch, seq_len, d_model]
            old_memory: Existing compressed memory [batch, mem_len, d_model] or None
            
        Returns:
            Updated compressed memory [batch, new_mem_len, d_model]
        """
        batch_size, new_len, _ = new_activations.shape
        
        # Combine old memory with new activations
        if old_memory is not None:
            combined = torch.cat([old_memory, new_activations], dim=1)
        else:
            combined = new_activations
        
        # If exceeds window, compress the oldest activations
        if combined.size(1) > self.window_size:
            # How many old activations to compress
            excess = combined.size(1) - self.window_size
            
            # Ensure we compress in multiples of compression_rate
            num_to_compress = (excess // self.compression_rate) * self.compression_rate
            
            if num_to_compress > 0:
                to_compress = combined[:, :num_to_compress, :]
                to_keep = combined[:, num_to_compress:, :]
                
                # Compress old activations (e.g., 512 -> 128)
                compressed = self.memory_manager.compress_activations(to_compress)
                
                # Concatenate: [compressed_old, recent]
                combined = torch.cat([compressed, to_keep], dim=1)
        
        # Limit total memory length to max_compressed_len
        if combined.size(1) > self.memory_manager.max_compressed_len:
            # Keep only the most recent max_compressed_len tokens
            combined = combined[:, -self.memory_manager.max_compressed_len:, :]
        
        # Detach to prevent backprop through entire history
        return combined.detach()
    
    def _flash_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Flash Attention 3 implementation using PyTorch's scaled_dot_product_attention.
        
        This provides significant speedup and memory reduction for long sequences.
        Uses automatic kernel selection by default for optimal performance.
        
        Args:
            q: Query [batch, n_heads, seq_len, head_dim]
            k: Key [batch, n_heads, kv_len, head_dim]
            v: Value [batch, n_heads, kv_len, head_dim]
            mask: Optional attention mask (for current sequence, not including memory)
            
        Returns:
            Attention output [batch, n_heads, seq_len, head_dim]
        """
        # Handle mask for compressive attention:
        # - Allow full attention to compressed memory (it's all in the past)
        # - Apply causal mask only within current sequence
        attn_mask = None
        
        if mask is not None and k.size(2) > q.size(2):
            # We have compressed memory: k/v include [memory, current_seq]
            batch_size, n_heads, seq_len, head_dim = q.shape
            kv_len = k.size(2)
            mem_len = kv_len - seq_len
            
            # Create extended mask: [batch, n_heads, seq_len, kv_len]
            # Structure: [can attend to all memory | causal mask for current seq]
            extended_mask = torch.zeros(
                batch_size, n_heads, seq_len, kv_len,
                dtype=q.dtype, device=q.device
            )
            
            # Part 1: Full attention to memory (all zeros = can attend)
            # extended_mask[:, :, :, :mem_len] = 0  # Already zeros
            
            # Part 2: Causal mask for current sequence
            if mask.size(-1) == seq_len:
                # Mask is for current sequence only: [batch, 1, seq_len, seq_len]
                # Expand and place in the current_seq part
                current_mask = mask.expand(batch_size, n_heads, seq_len, seq_len)
                extended_mask[:, :, :, mem_len:] = current_mask
            
            attn_mask = extended_mask
            # Convert 0s to allow attention, large negative to mask
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float('-inf'))
            
        elif mask is not None:
            # No compressed memory, just use mask as-is
            if mask.dim() == 4 and mask.size(1) == 1:
                attn_mask = mask.expand(-1, self.n_heads, -1, -1)
            else:
                attn_mask = mask
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float('-inf'))
        
        # Configure backend
        enable_flash = self.flash_backend in ['flash', 'auto']
        enable_mem_efficient = self.flash_backend in ['mem_efficient', 'auto']
        enable_math = self.flash_backend in ['math', 'auto']
        
        with torch.backends.cuda.sdp_kernel(
            enable_flash=enable_flash,
            enable_math=enable_math,
            enable_mem_efficient=enable_mem_efficient
        ):
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout_p if self.training else 0.0,
                is_causal=False  # We handle causality with explicit mask
            )
        
        return attn_output
    
    def _standard_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Standard scaled dot-product attention (fallback implementation).
        
        Args:
            q: Query [batch, n_heads, seq_len, head_dim]
            k: Key [batch, n_heads, kv_len, head_dim]
            v: Value [batch, n_heads, kv_len, head_dim]
            mask: Optional attention mask (for current sequence, not including memory)
            
        Returns:
            Attention output [batch, n_heads, seq_len, head_dim]
        """
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Handle mask for compressive attention (same logic as _flash_attention)
        if mask is not None and k.size(2) > q.size(2):
            # We have compressed memory
            batch_size, n_heads, seq_len, head_dim = q.shape
            kv_len = k.size(2)
            mem_len = kv_len - seq_len
            
            # Create extended mask
            extended_mask = torch.ones(
                batch_size, n_heads, seq_len, kv_len,
                dtype=torch.bool, device=q.device
            )
            
            # Allow attention to all memory
            extended_mask[:, :, :, :mem_len]  = True
            
            # Apply causal mask to current sequence
            if mask.size(-1) == seq_len:
                current_mask = mask.expand(batch_size, n_heads, seq_len, seq_len)
                extended_mask[:, :, :, mem_len:] = current_mask.bool()
            
            attn_scores = attn_scores.masked_fill(~extended_mask, float('-inf'))
            
        elif mask is not None:
            # No compressed memory
            if mask.dim() == 4 and mask.size(1) == 1:
                mask = mask.expand(-1, self.n_heads, -1, -1)
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        
        return attn_output
    
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
        use_flash_attention: Enable Flash Attention for faster inference
        flash_backend: Flash attention backend selection
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_ff: int = 2048,
        window_size: int = 512,
        compression_rate: int = 4,
        dropout: float = 0.1,
        use_flash_attention: bool = True,
        flash_backend: Literal['flash', 'mem_efficient', 'math', 'auto'] = 'auto'
    ):
        super().__init__()
        
        self.attention = CompressiveAttention(
            d_model=d_model,
            n_heads=n_heads,
            window_size=window_size,
            compression_rate=compression_rate,
            dropout=dropout,
            use_flash_attention=use_flash_attention,
            flash_backend=flash_backend
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
