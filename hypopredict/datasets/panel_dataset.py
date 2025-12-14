"""
PyTorch Dataset for time-series panel data.

Provides clean integration between chunked sensor data and PyTorch models.
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Union
import torch
from torch.utils.data import Dataset


class PanelDataset(Dataset):
    """
    PyTorch Dataset for panel time-series data (multiple persons, multiple days).
    
    Handles ECG/sensor chunks with associated labels for hypoglycemia prediction.
    
    Attributes:
        chunks: List of DataFrames containing sensor data chunks
        labels: List of binary labels (0=no HG, 1=HG event predicted)
        person_days: Optional list of PersonDay objects for tracking
    """
    
    def __init__(
        self,
        chunks: List[pd.DataFrame],
        labels: List[int],
        person_days: Optional[List] = None,
        transform: Optional[callable] = None
    ):
        """
        Initialize PanelDataset.
        
        Args:
            chunks: List of DataFrames with sensor data (datetime-indexed)
            labels: List of binary labels (0 or 1) for each chunk
            person_days: Optional list of PersonDay identifiers for each chunk
            transform: Optional transform to apply to each chunk
        """
        if len(chunks) != len(labels):
            raise ValueError(
                f"Number of chunks ({len(chunks)}) must match "
                f"number of labels ({len(labels)})"
            )
        
        self.chunks = chunks
        self.labels = labels
        self.person_days = person_days
        self.transform = transform
        
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.chunks)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (chunk_tensor, label)
        """
        chunk = self.chunks[idx]
        label = self.labels[idx]
        
        # Convert DataFrame to numpy array
        chunk_array = chunk.values
        
        # Apply transform if provided
        if self.transform:
            chunk_array = self.transform(chunk_array)
        
        # Convert to PyTorch tensor
        chunk_tensor = torch.FloatTensor(chunk_array)
        
        return chunk_tensor, label
    
    def get_person_day(self, idx: int) -> Optional[Union[int, 'PersonDay']]:
        """
        Get the PersonDay identifier for a sample.
        
        Args:
            idx: Index of the sample
            
        Returns:
            PersonDay object or legacy int identifier, or None if not available
        """
        if self.person_days is not None:
            return self.person_days[idx]
        return None
    
    @staticmethod
    def collate_fn(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Custom collate function for DataLoader.
        
        Handles variable-length sequences by padding to max length in batch.
        
        Args:
            batch: List of (chunk_tensor, label) tuples
            
        Returns:
            Tuple of (batched_chunks, batched_labels)
        """
        chunks, labels = zip(*batch)
        
        # Find max length in batch
        max_len = max(chunk.shape[0] for chunk in chunks)
        num_features = chunks[0].shape[1] if len(chunks[0].shape) > 1 else 1
        
        # Pad chunks to max length
        padded_chunks = []
        for chunk in chunks:
            if len(chunk.shape) == 1:
                chunk = chunk.unsqueeze(1)
            
            pad_len = max_len - chunk.shape[0]
            if pad_len > 0:
                padding = torch.zeros(pad_len, chunk.shape[1])
                chunk = torch.cat([chunk, padding], dim=0)
            
            padded_chunks.append(chunk)
        
        batched_chunks = torch.stack(padded_chunks)
        batched_labels = torch.LongTensor(labels)
        
        return batched_chunks, batched_labels
