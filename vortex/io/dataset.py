"""
Binary dataset loader with sliding-window support for byte-level compression.

This module provides efficient loading and preprocessing of raw binary data
for neural compression systems. It supports arbitrary binary formats including
industrial telemetry logs, sensor data dumps, and structured binary files.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Union, Optional


class ByteDataset(Dataset):
    """
    Dataset for loading raw binary data as uint8 byte streams.
    
    Provides sliding-window access to binary files for training neural
    compression models. Each sample represents a consecutive sequence of
    bytes that can be used for next-byte prediction tasks.
    
    Args:
        file_path: Path to binary file (.log, .dat, or any binary format)
        window_size: Length of byte sequence for each sample
        stride: Step size between consecutive windows (default: window_size for non-overlapping)
        max_bytes: Optional limit on total bytes to load from file
        
    Example:
        >>> dataset = ByteDataset("telemetry.dat", window_size=512, stride=256)
        >>> sequence = dataset[0]  # Returns tensor of shape [512] with uint8 values
    """
    
    def __init__(
        self,
        file_path: Union[str, Path],
        window_size: int,
        stride: Optional[int] = None,
        max_bytes: Optional[int] = None
    ):
        self.file_path = Path(file_path)
        self.window_size = window_size
        self.stride = stride if stride is not None else window_size
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"Binary file not found: {self.file_path}")
        
        with open(self.file_path, "rb") as f:
            raw_data = f.read(max_bytes) if max_bytes else f.read()
        
        self.data = np.frombuffer(raw_data, dtype=np.uint8)
        
        if len(self.data) < window_size:
            raise ValueError(
                f"File contains {len(self.data)} bytes, but window_size={window_size}. "
                "Cannot create valid samples."
            )
        
        self.num_samples = max(1, (len(self.data) - window_size) // self.stride + 1)
    
    def __len__(self) -> int:
        """Returns the total number of sliding-window samples available."""
        return self.num_samples
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Retrieves a single window of bytes from the dataset.
        
        Args:
            idx: Index of the window to retrieve
            
        Returns:
            Tensor of shape [window_size] containing uint8 byte values
        """
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range for dataset with {self.num_samples} samples")
        
        start_idx = idx * self.stride
        end_idx = start_idx + self.window_size
        
        if end_idx > len(self.data):
            end_idx = len(self.data)
            start_idx = max(0, end_idx - self.window_size)
        
        window = self.data[start_idx:end_idx]
        return torch.from_numpy(window.copy()).long()
    
    def get_raw_bytes(self) -> np.ndarray:
        """Returns the full raw byte array for direct access."""
        return self.data
    
    @property
    def total_bytes(self) -> int:
        """Returns the total number of bytes loaded from file."""
        return len(self.data)
    
    @property
    def vocab_size(self) -> int:
        """Returns the vocabulary size (always 256 for byte-level data)."""
        return 256


class StreamingByteDataset:
    """
    Memory-efficient streaming dataset for very large binary files.
    
    Instead of loading the entire file into memory, this class uses
    memory-mapped I/O to access byte sequences on-demand. Suitable for
    production deployment on multi-terabyte telemetry archives.
    
    Args:
        file_path: Path to binary file
        window_size: Length of byte sequence for each sample
        stride: Step size between consecutive windows
    """
    
    def __init__(
        self,
        file_path: Union[str, Path],
        window_size: int,
        stride: Optional[int] = None
    ):
        self.file_path = Path(file_path)
        self.window_size = window_size
        self.stride = stride if stride is not None else window_size
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"Binary file not found: {self.file_path}")
        
        self.file_size = self.file_path.stat().st_size
        
        if self.file_size < window_size:
            raise ValueError(
                f"File size ({self.file_size} bytes) smaller than window_size ({window_size})"
            )
        
        self.mmap = np.memmap(self.file_path, dtype=np.uint8, mode='r')
        self.num_samples = max(1, (self.file_size - window_size) // self.stride + 1)
    
    def __len__(self) -> int:
        """Returns the total number of sliding-window samples available."""
        return self.num_samples
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Retrieves a single window of bytes using memory-mapped I/O.
        
        Args:
            idx: Index of the window to retrieve
            
        Returns:
            Tensor of shape [window_size] containing uint8 byte values
        """
        if idx < 0 or idx >= self.num_samples:
            raise IndexError(f"Index {idx} out of range for dataset with {self.num_samples} samples")
        
        start_idx = idx * self.stride
        end_idx = start_idx + self.window_size
        
        if end_idx > self.file_size:
            end_idx = self.file_size
            start_idx = max(0, end_idx - self.window_size)
        
        window = np.array(self.mmap[start_idx:end_idx])
        return torch.from_numpy(window).long()
    
    @property
    def total_bytes(self) -> int:
        """Returns the total file size in bytes."""
        return self.file_size
    
    def close(self):
        """Closes the memory-mapped file handle."""
        if hasattr(self, 'mmap'):
            del self.mmap
