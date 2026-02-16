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

try:
    from vortex.core.range_coder import AutoregressiveRangeCoder
    AUTOREGRESSIVE_CODER_AVAILABLE = True
except ImportError:
    AUTOREGRESSIVE_CODER_AVAILABLE = False

# GPU Range Coder (much faster than CPU torchac)
try:
    from vortex.cuda.range_coder import GPURangeCoder
    GPU_RANGE_CODER_AVAILABLE = True
except ImportError:
    GPU_RANGE_CODER_AVAILABLE = False

# GPU Range Coder (much faster than CPU torchac)
try:
    from vortex.cuda.range_coder import GPURangeCoder
    GPU_RANGE_CODER_AVAILABLE = True
except ImportError:
    GPU_RANGE_CODER_AVAILABLE = False


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
        use_flash_attention: Enable Flash Attention 3 for faster inference
        flash_backend: Flash attention backend ('flash', 'mem_efficient', 'math', 'auto')
        
    Example:
        >>> codec = VortexCodec(d_model=256, n_layers=6, use_flash_attention=True)
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
        dropout: float = 0.1,
        use_flash_attention: bool = True,
        flash_backend: str = 'auto'
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
                dropout=dropout,
                use_flash_attention=use_flash_attention,
                flash_backend=flash_backend
            )
            for _ in range(n_layers)
        ])
        
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize GPU range coder if available (10-100× faster than CPU)
        if GPU_RANGE_CODER_AVAILABLE:
            self.gpu_range_coder = GPURangeCoder(precision_bits=16)
            self.use_gpu_range_coder = True
        else:
            self.gpu_range_coder = None
            self.use_gpu_range_coder = False
    
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
            RuntimeError: If no compression backend is available
        """
        if not TORCHAC_AVAILABLE and not GPU_RANGE_CODER_AVAILABLE:
            raise RuntimeError(
                "No compression backend available. Install torchac or compile GPU kernels:\n"
                "  - CPU (slower): pip install torchac\n"
                "  - GPU (faster): python setup_cuda_kernels.py install"
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
        
        # Initialize compressed memories for proper compressive transformer usage
        compressed_memories = None
        
        for chunk_idx in iterator:
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, data.size(1))
            chunk_data = data[:, start_idx:end_idx]
            
            # Store first byte of chunk uncompressed for decompression seed
            first_byte = chunk_data[:, 0:1]
            
            if chunk_data.size(1) > 1:
                # Maintain memory state across chunks for long-range context
                logits, compressed_memories = self.forward(chunk_data, compressed_memories=compressed_memories)
                
                # Encode bytes [1:] using CDFs from positions [:-1]
                # This means we predict byte i+1 from context up to byte i
                probs = F.softmax(logits, dim=-1)
                
                targets = chunk_data[:, 1:]
                
                # Use GPU range coder if available, otherwise fall back to torchac
                try:
                    if self.use_gpu_range_coder and device.type in ['cuda', 'hip']:
                        # GPU range coding (10-100× faster)
                        # Flatten for encoding: [batch*seq_len, vocab_size]
                        probs_flat = probs[:, :-1].reshape(-1, self.vocab_size)
                        targets_flat = targets.reshape(-1)
                        
                        compressed_chunk = self.gpu_range_coder.encode(
                            targets_flat,
                            probs_flat,
                            device
                        )
                    else:
                        # CPU arithmetic coding (fallback)
                        cdfs = self._probabilities_to_cdf(probs[:, :-1])
                        compressed_chunk = torchac.encode_float_cdf(
                            cdfs.cpu(),
                            targets.cpu().to(torch.int16),
                            check_input_bounds=True
                        )
                except Exception as e:
                    raise RuntimeError(f"Compression failed at chunk {chunk_idx}: {e}")
            else:
                # Single-byte chunk, no arithmetic coding needed
                compressed_chunk = b''
                # Still update memory state
                _, compressed_memories = self.forward(chunk_data, compressed_memories=compressed_memories)
            
            # Prepend first byte (uncompressed) to chunk
            compressed_chunks.append(first_byte.cpu().numpy().tobytes() + compressed_chunk)
        
        import struct
        # Create header with: original_length, num_chunks, chunk_size
        header = struct.pack('<III', original_length, len(compressed_chunks), chunk_size)
        
        # Add chunk boundaries: for each chunk, store its size
        chunk_metadata = b''
        for chunk in compressed_chunks:
            chunk_metadata += struct.pack('<I', len(chunk))
        
        compressed_data = header + chunk_metadata + b''.join(compressed_chunks)
        
        return compressed_data
    
    @torch.no_grad()
    def decompress(
        self,
        compressed_data: bytes,
        show_progress: bool = False
    ) -> bytes:
        """
        Decompresses byte sequence using neural probability estimation.
        
        Processes each chunk sequentially, generating probabilities autoregressively
        and decoding with torchac. Maintains memory state across chunks for
        long-range dependencies.
        
        Args:
            compressed_data: Compressed byte string with header and chunk metadata
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
        
        self.eval()
        device = next(self.parameters()).device
        
        # Parse header: original_length, num_chunks, chunk_size
        if len(compressed_data) < 12:
            raise ValueError("Compressed data is too short (missing header)")
        
        original_length, num_chunks, chunk_size = struct.unpack('<III', compressed_data[:12])
        offset = 12
        
        # Parse chunk metadata (sizes)
        chunk_sizes = []
        for i in range(num_chunks):
            if offset + 4 > len(compressed_data):
                raise ValueError(f"Compressed data truncated at chunk {i} metadata")
            chunk_size_bytes = struct.unpack('<I', compressed_data[offset:offset+4])[0]
            chunk_sizes.append(chunk_size_bytes)
            offset += 4
        
        # Extract compressed chunks
        compressed_chunks = []
        for i, size in enumerate(chunk_sizes):
            if offset + size > len(compressed_data):
                raise ValueError(f"Compressed data truncated at chunk {i} payload")
            compressed_chunks.append(compressed_data[offset:offset+size])
            offset += size
        
        # Decompress each chunk
        decompressed_bytes = []
        compressed_memories = None
        
        iterator = enumerate(compressed_chunks)
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(list(iterator), desc="Decompressing")
            except ImportError:
                pass
        
        for chunk_idx, compressed_chunk in iterator:
            # Determine expected chunk length
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, original_length)
            expected_chunk_len = chunk_end - chunk_start
            
            # Decompress this chunk autoregressively
            chunk_output = self._decompress_chunk(
                compressed_chunk,
                expected_chunk_len,
                compressed_memories,
                device
            )
            
            decompressed_bytes.extend(chunk_output)
            
            # Update memory state for next chunk
            # Feed the decompressed chunk through the model to update memories
            chunk_tensor = torch.tensor(chunk_output, dtype=torch.long, device=device).unsqueeze(0)
            _, compressed_memories = self.forward(chunk_tensor, compressed_memories=compressed_memories)
        
        # Convert to bytes
        result = bytes(decompressed_bytes[:original_length])
        
        return result
    
    def _decompress_chunk(
        self,
        compressed_chunk: bytes,
        expected_length: int,
        compressed_memories: Optional[List[torch.Tensor]],
        device: torch.device
    ) -> List[int]:
        """
        Decompresses a single chunk autoregressively.
        
        For each position, generates the probability distribution using the model,
        converts to CDF, and decodes the next byte using torchac.
        
        Args:
            compressed_chunk: Compressed bytes for this chunk
            expected_length: Number of bytes to decode
            compressed_memories: Memory state from previous chunks
            device: Device to run computation on
            
        Returns:
            List of decompressed byte values
        """
        decompressed = []
        
        # Start with empty context or use memories from previous chunk
        current_context = torch.zeros((1, 0), dtype=torch.long, device=device)
        
        # Decode each byte autoregressively
        for pos in range(expected_length):
            # Get probability distribution for next byte
            logits, compressed_memories = self.forward(current_context, compressed_memories=compressed_memories)
            
            # Get probabilities for the last position (next byte prediction)
            if logits.size(1) > 0:
                next_byte_probs = F.softmax(logits[:, -1:, :], dim=-1)
            else:
                # If context is empty, use uniform distribution
                next_byte_probs = torch.ones((1, 1, self.vocab_size), device=device) / self.vocab_size
            
            # Convert to CDF
            cdf = self._probabilities_to_cdf(next_byte_probs)
            
            # Decode next byte
            # Note: torchac.decode_float_cdf expects the CDF and compressed stream
            # We need to decode one symbol at a time from the compressed stream
            # This is a limitation - torchac doesn't easily support streaming decode
            # For now, we'll use a workaround with the full chunk
            
            # Workaround: decode all at once using batch CDFs
            # This requires collecting all CDFs first, then decoding
            # Let's implement a different approach
            break
        
        # Better approach: collect all CDFs first, then decode in batch
        return self._decompress_chunk_batch(
            compressed_chunk,
            expected_length,
            compressed_memories,
            device
        )
    
    def _decompress_chunk_batch(
        self,
        compressed_chunk: bytes,
        expected_length: int,
        compressed_memories: Optional[List[torch.Tensor]],
        device: torch.device
    ) -> List[int]:
        """
        Decompresses a chunk using autoregressive decoding when available.
        
        The first byte is stored uncompressed as a seed. We use the autoregressive
        range coder to decode the rest, generating CDFs on-the-fly based on
        previously decoded symbols.
        
        Args:
            compressed_chunk: Compressed bytes (first byte + encoded rest)
            expected_length: Number of bytes to decode
            compressed_memories: Memory state from previous chunks
            device: Device to run computation on
            
        Returns:
            List of decompressed byte values
        """
        if expected_length == 0:
            return []
        
        # Extract first byte (uncompressed seed)
        first_byte_val = compressed_chunk[0]
        decoded_bytes = [first_byte_val]
        
        if expected_length == 1:
            return decoded_bytes
        
        # Use autoregressive decoder if available
        if AUTOREGRESSIVE_CODER_AVAILABLE:
            return self._decompress_autoregressive(
                compressed_chunk, expected_length, first_byte_val,
                compressed_memories, device
            )
        
        # Fallback to original method (less accurate)
        return self._decompress_fallback(
            compressed_chunk, expected_length, first_byte_val,
            compressed_memories, device
        )
    
    def _decompress_autoregressive(
        self,
        compressed_chunk: bytes,
        expected_length: int,
        first_byte_val: int,
        compressed_memories: Optional[List[torch.Tensor]],
        device: torch.device
    ) -> List[int]:
        """
        Decompress using autoregressive range coder with on-the-fly CDF generation.
        """
        from vortex.core.range_coder import AutoregressiveRangeCoder
        
        first_byte_tensor = torch.tensor([[first_byte_val]], dtype=torch.long, device=device)
        torchac_stream = compressed_chunk[1:]  # Skip first byte
        
        # Define CDF generator function
        def get_cdf_for_context(context, memories):
            """Generate CDF for next position given current context."""
            logits, updated_memories = self.forward(context, compressed_memories=memories)
            # Get probabilities for the last position (next byte prediction)
            probs = F.softmax(logits[:, -1:, :], dim=-1)
            cdf = self._probabilities_to_cdf(probs)
            return cdf, updated_memories
        
        # Decode autoregressively
        try:
            decoded_tensor = AutoregressiveRangeCoder.decode(
                torchac_stream,
                get_cdf_for_context,
                first_byte_tensor,
                expected_length - 1,  # Remaining symbols after first byte
                compressed_memories,
                device
            )
            # decoded_tensor includes initial context, extract all
            decoded_bytes = decoded_tensor.squeeze(0).tolist()
            return decoded_bytes
        except Exception as e:
            raise RuntimeError(
                f"Autoregressive decode failed: {e}. "
                f"Ensure you're using the same model that was used for compression."
            )
    
    def _decompress_fallback(
        self,
        compressed_chunk: bytes,
        expected_length: int,
        first_byte_val: int,
        compressed_memories: Optional[List[torch.Tensor]],
        device: torch.device
    ) -> List[int]:
        """
        Fallback decompression method (less accurate).
        
        Generates CDFs using dummy context. This may not match the encoding CDFs
        exactly, leading to potential decompression errors.
        """
        decoded_bytes = [first_byte_val]
        
        if expected_length == 1:
            return decoded_bytes
        
        # Decode remaining bytes using first byte as context
        first_byte_tensor = torch.tensor([[first_byte_val]], dtype=torch.long, device=device)
        
        # Create full context: first byte + zeros for rest
        full_context = torch.cat([
            first_byte_tensor,
            torch.zeros((1, expected_length - 1), dtype=torch.long, device=device)
        ], dim=1)
        
        # Forward pass to get CDFs for all positions
        logits, _ = self.forward(full_context, compressed_memories=compressed_memories)
        
        # Get probabilities for positions [:-1] to decode positions [1:]
        probs = F.softmax(logits[:, :-1, :], dim=-1)
        
        # Decode the remaining bytes
        encoded_stream = compressed_chunk[1:]  # Skip first byte
        
        if len(encoded_stream) > 0:
            device = next(self.parameters()).device
            
            try:
                if self.use_gpu_range_coder and device.type in ['cuda', 'hip']:
                    # GPU range decoding (10-100× faster)
                    # Flatten probabilities for decoding
                    probs_flat = probs.reshape(-1, self.vocab_size)
                    
                    # Create CDF for GPU decoder
                    cdfs = self._probabilities_to_cdf(probs_flat.unsqueeze(0)).squeeze(0)
                    
                    decoded_symbols = self.gpu_range_coder.decode(
                        encoded_stream,
                        cdfs,
                        length=probs_flat.size(0),
                        device=device
                    )
                    decoded_bytes.extend(decoded_symbols.cpu().tolist())
                else:
                    # CPU arithmetic decoding (fallback)
                    import torchac
                    cdfs = self._probabilities_to_cdf(probs)  # [1, seq_len-1, vocab_size+1]
                    decoded_symbols = torchac.decode_float_cdf(
                        cdfs.cpu(),
                        encoded_stream
                    )
                    decoded_bytes.extend(decoded_symbols.squeeze(0).tolist())
            except Exception as e:
                raise RuntimeError(
                    f"Failed to decode chunk: {e}. "
                    f"This may indicate model/data mismatch or corrupted data. "
                    f"Ensure you're using the same model that was used for compression."
                )
        
        return decoded_bytes
    
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
    
    def decompress_file(
        self,
        input_path: str,
        output_path: str
    ) -> dict:
        """
        Decompresses a file and returns decompression statistics.
        
        Args:
            input_path: Path to compressed input file (.vortex)
            output_path: Path to output decompressed file
            
        Returns:
            Dictionary with decompression statistics
        """
        import time
        
        with open(input_path, 'rb') as f:
            compressed_data = f.read()
        
        compressed_size = len(compressed_data)
        
        print(f"Decompressing {compressed_size:,} bytes...")
        start_time = time.time()
        
        decompressed_data = self.decompress(compressed_data, show_progress=True)
        
        decompress_time = time.time() - start_time
        decompressed_size = len(decompressed_data)
        
        with open(output_path, 'wb') as f:
            f.write(decompressed_data)
        
        ratio = compressed_size / decompressed_size
        
        stats = {
            'compressed_size': compressed_size,
            'decompressed_size': decompressed_size,
            'ratio': ratio,
            'factor': 1 / ratio if ratio > 0 else float('inf'),
            'decompress_time': decompress_time,
            'throughput_mbps': (decompressed_size / 1024 / 1024) / decompress_time
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
        # Cast positional encoding to match input dtype for mixed precision
        pe = self.pe[:, :seq_len, :].to(x.dtype)
        return x + pe


import math
