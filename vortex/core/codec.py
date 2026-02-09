"""
High-level compression and decompression interface using arithmetic coding.

This module provides the VortexCodec class that orchestrates the complete
compression pipeline: neural probability estimation via Compressive Transformer,
conversion to CDFs, and lossless encoding via torchac arithmetic coder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Union, Tuple, List
from pathlib import Path

try:
    import torchac
    TORCHAC_AVAILABLE = True
except ImportError:
    TORCHAC_AVAILABLE = False
    import warnings
    warnings.warn(
        "torchac not available. Install with: pip install torchac\n"
        "Codec will run in probability-estimation-only mode."
    )


from vortex.modules.compressive import CompressiveTransformerBlock


class VortexCodec(nn.Module):
    """
    Neural lossless codec using Compressive Transformer for probability estimation.
    
    This codec predicts a probability distribution over the next byte (0-255) given
    previous context. The predicted distribution is converted to a cumulative
    distribution function (CDF) and passed to an arithmetic coder for lossless
    compression. The compressive architecture allows modeling of very long-range
    dependencies in structured binary data.
    
    Args:
        d_model: Transformer hidden dimension
        n_layers: Number of transformer blocks
        n_heads: Number of attention heads per block
        d_ff: Feed-forward network hidden dimension
        window_size: Size of recent context window
        compression_rate: Ratio for compressing old activations
        vocab_size: Output vocabulary size (256 for bytes)
        max_seq_len: Maximum sequence length supported
        dropout: Dropout probability
        
    Example:
        >>> codec = VortexCodec(d_model=256, n_layers=6)
        >>> compressed = codec.compress(input_bytes)
        >>> decompressed = codec.decompress(compressed, target_length=len(input_bytes))
    """
    
    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 1024,
        window_size: int = 512,
        compression_rate: int = 4,
        vocab_size: int = 256,
        max_seq_len: int = 8192,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_layers = n_layers
        self.window_size = window_size
        self.compression_rate = compression_rate
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        
        self.byte_embedding = nn.Embedding(vocab_size, d_model)
        
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_seq_len)
        
        self.transformer_blocks = nn.ModuleList([
            CompressiveTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                window_size=window_size,
                compression_rate=compression_rate,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])
        
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        compressed_memories: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass to predict next-byte probabilities.
        
        Args:
            x: Input byte sequence [batch_size, seq_len] with values in [0, 255]
            compressed_memories: Optional list of compressed memory buffers per layer
            
        Returns:
            Tuple of (logits, updated_memories)
            - logits: [batch, seq_len, vocab_size] unnormalized predictions
            - updated_memories: List of memory buffers for each layer
        """
        batch_size, seq_len = x.shape
        
        x = self.byte_embedding(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        if compressed_memories is None:
            compressed_memories = [None] * self.n_layers
        
        updated_memories = []
        
        for i, block in enumerate(self.transformer_blocks):
            x, updated_memory = block(
                x,
                compressed_memory=compressed_memories[i],
                mask=self._create_causal_mask(seq_len, x.device)
            )
            updated_memories.append(updated_memory)
        
        logits = self.output_projection(x)
        
        return logits, updated_memories
    
    def predict_distribution(
        self,
        context: torch.Tensor,
        compressed_memories: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Predicts probability distribution over next byte given context.
        
        Args:
            context: Previous bytes [batch_size, context_len]
            compressed_memories: Optional memory buffers
            
        Returns:
            Tuple of (probabilities, updated_memories)
            - probabilities: [batch, vocab_size] normalized probability distribution
            - updated_memories: Updated memory buffers
        """
        logits, updated_memories = self.forward(context, compressed_memories)
        
        next_byte_logits = logits[:, -1, :]
        probabilities = F.softmax(next_byte_logits, dim=-1)
        
        return probabilities, updated_memories
    
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Creates causal attention mask to prevent attending to future positions.
        
        Args:
            seq_len: Sequence length
            device: Target device for tensor
            
        Returns:
            Boolean mask of shape [1, 1, seq_len, seq_len]
        """
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask == 0
        return mask.unsqueeze(0).unsqueeze(0)
    
    @torch.no_grad()
    def compress(
        self,
        data: Union[torch.Tensor, np.ndarray, bytes],
        chunk_size: int = 256,
        show_progress: bool = False
    ) -> bytes:
        """
        Compresses input byte sequence using neural probability estimation.
        
        Uses chunk-based processing for efficiency. Each chunk is compressed
        independently with arithmetic coding based on predicted CDFs.
        
        Args:
            data: Input data as tensor, numpy array, or raw bytes
            chunk_size: Size of chunks to process at once
            show_progress: Whether to show progress bar
            
        Returns:
            Compressed byte string (header + compressed chunks)
            
        Raises:
            RuntimeError: If torchac is not available
        """
        if not TORCHAC_AVAILABLE:
            raise RuntimeError(
                "torchac is required for compression. Install with: pip install torchac"
            )
        
        self.eval()
        
        if isinstance(data, bytes):
            data = np.frombuffer(data, dtype=np.uint8).copy()
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data.copy()).long()
        
        original_length = len(data) if data.dim() == 1 else data.size(1)
        
        if data.dim() == 1:
            data = data.unsqueeze(0)
        
        device = next(self.parameters()).device
        data = data.to(device)
        
        num_chunks = (data.size(1) + chunk_size - 1) // chunk_size
        compressed_chunks = []
        
        iterator = range(num_chunks)
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc="Compressing")
            except ImportError:
                pass
        
        for chunk_idx in iterator:
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, data.size(1))
            chunk_data = data[:, start_idx:end_idx]
            
            logits, _ = self.forward(chunk_data)
            
            probs = F.softmax(logits, dim=-1)
            cdfs = self._probabilities_to_cdf(probs[:, :-1])
            
            targets = chunk_data[:, 1:]
            
            try:
                compressed_chunk = torchac.encode_float_cdf(
                    cdfs.cpu(),
                    targets.cpu().to(torch.int16),
                    check_input_bounds=True
                )
                compressed_chunks.append(compressed_chunk)
            except Exception as e:
                raise RuntimeError(f"Compression failed at chunk {chunk_idx}: {e}")
        
        import struct
        header = struct.pack('<I', original_length)
        compressed_data = header + b''.join(compressed_chunks)
        
        return compressed_data
    
    @torch.no_grad()
    def decompress(
        self,
        compressed_data: bytes,
        chunk_size: int = 256,
        show_progress: bool = False
    ) -> bytes:
        """
        Decompresses byte sequence using neural probability estimation.
        
        Note: This is a simplified implementation. Full streaming decompression
        requires maintaining state between chunks and careful synchronization
        with the compression process.
        
        Args:
            compressed_data: Compressed byte string with header
            chunk_size: Size of chunks used during compression
            show_progress: Whether to show progress bar
            
        Returns:
            Decompressed byte array
            
        Raises:
            RuntimeError: If torchac is not available or decompression fails
        """
        if not TORCHAC_AVAILABLE:
            raise RuntimeError(
                "torchac is required for decompression. Install with: pip install torchac"
            )
        
        import struct
        
        if len(compressed_data) < 4:
            raise ValueError("Compressed data is too short (missing header)")
        
        original_length = struct.unpack('<I', compressed_data[:4])[0]
        compressed_payload = compressed_data[4:]
        
        raise NotImplementedError(
            "Full decompression requires chunk boundary markers in the compressed stream. "
            "This is a limitation of the current implementation. "
            "For now, use the model for compression evaluation only (BPD measurement)."
        )
    
    def _probabilities_to_cdf(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Converts probability distribution to cumulative distribution function.
        
        Args:
            probs: Probability distribution [..., vocab_size]
            
        Returns:
            CDF tensor [..., vocab_size + 1]
        """
        cdf = torch.cumsum(probs, dim=-1)
        
        zeros_shape = list(cdf.shape)
        zeros_shape[-1] = 1
        zeros = torch.zeros(zeros_shape, dtype=cdf.dtype, device=cdf.device)
        
        cdf = torch.cat([zeros, cdf], dim=-1)
        
        cdf = torch.clamp(cdf, 0.0, 1.0)
        
        return cdf
    
    def compress_file(
        self,
        input_path: str,
        output_path: str,
        chunk_size: int = 256,
        max_bytes: Optional[int] = None
    ) -> dict:
        """
        Compresses a file and returns compression statistics.
        
        Args:
            input_path: Path to input file
            output_path: Path to output compressed file
            chunk_size: Chunk size for compression
            max_bytes: Optional limit on bytes to compress
            
        Returns:
            Dictionary with compression statistics
        """
        import time
        
        with open(input_path, 'rb') as f:
            raw_data = f.read(max_bytes) if max_bytes else f.read()
        
        original_size = len(raw_data)
        
        print(f"Compressing {original_size:,} bytes...")
        start_time = time.time()
        
        compressed_data = self.compress(raw_data, chunk_size=chunk_size, show_progress=True)
        
        compress_time = time.time() - start_time
        compressed_size = len(compressed_data)
        
        with open(output_path, 'wb') as f:
            f.write(compressed_data)
        
        ratio = compressed_size / original_size
        bpd = (compressed_size * 8) / original_size
        
        stats = {
            'original_size': original_size,
            'compressed_size': compressed_size,
            'ratio': ratio,
            'factor': 1 / ratio if ratio > 0 else float('inf'),
            'savings_pct': (1 - ratio) * 100,
            'bpd': bpd,
            'compress_time': compress_time,
            'throughput_mbps': (original_size / 1024 / 1024) / compress_time
        }
        
        return stats
    
    def save(self, path: Union[str, Path]):
        """Saves model weights to disk."""
        torch.save(self.state_dict(), path)
    
    def load(self, path: Union[str, Path]):
        """Loads model weights from disk."""
        state_dict = torch.load(path, map_location='cpu')
        self.load_state_dict(state_dict)


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer inputs.
    
    Adds position information to byte embeddings using sine and cosine
    functions at different frequencies, allowing the model to distinguish
    byte positions in the sequence.
    
    Args:
        d_model: Embedding dimension
        max_len: Maximum sequence length to pre-compute encodings for
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding to input embeddings.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Input with positional encoding added [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


import math
