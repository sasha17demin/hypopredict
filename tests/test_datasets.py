"""
Tests for PyTorch datasets module.
"""
import unittest
import pandas as pd
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from hypopredict.core.data_types import PersonDay


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch not available")
class TestPanelDataset(unittest.TestCase):
    """Test PanelDataset for PyTorch integration."""
    
    def setUp(self):
        """Set up test data."""
        from hypopredict.datasets.panel_dataset import PanelDataset
        self.PanelDataset = PanelDataset
        
        # Create mock chunks
        self.chunks = []
        self.labels = []
        self.person_days = []
        
        for i in range(10):
            dates = pd.date_range('2023-01-01', periods=100, freq='4ms')
            chunk = pd.DataFrame({
                'EcgWaveform': np.random.randn(100)
            }, index=dates)
            self.chunks.append(chunk)
            self.labels.append(i % 2)  # Alternate 0, 1
            self.person_days.append(PersonDay(i % 3 + 1, i % 6 + 1))
    
    def test_dataset_creation(self):
        """Test creating a PanelDataset."""
        dataset = self.PanelDataset(self.chunks, self.labels)
        self.assertEqual(len(dataset), len(self.chunks))
    
    def test_dataset_length_mismatch(self):
        """Test that mismatched lengths raise ValueError."""
        with self.assertRaises(ValueError):
            self.PanelDataset(self.chunks, [0, 1])  # Too few labels
    
    def test_getitem(self):
        """Test getting items from dataset."""
        dataset = self.PanelDataset(self.chunks, self.labels)
        
        chunk_tensor, label = dataset[0]
        
        # Check types
        self.assertIsInstance(chunk_tensor, torch.Tensor)
        self.assertIsInstance(label, int)
        
        # Check values
        self.assertEqual(label, self.labels[0])
    
    def test_person_days(self):
        """Test storing and retrieving PersonDay information."""
        dataset = self.PanelDataset(self.chunks, self.labels, self.person_days)
        
        pd_retrieved = dataset.get_person_day(0)
        self.assertEqual(pd_retrieved, self.person_days[0])
    
    def test_collate_fn(self):
        """Test custom collate function."""
        dataset = self.PanelDataset(self.chunks, self.labels)
        
        # Create a batch
        batch = [dataset[i] for i in range(3)]
        
        # Collate
        batched_chunks, batched_labels = self.PanelDataset.collate_fn(batch)
        
        # Check shapes
        self.assertIsInstance(batched_chunks, torch.Tensor)
        self.assertIsInstance(batched_labels, torch.Tensor)
        self.assertEqual(batched_labels.shape[0], 3)


if __name__ == '__main__':
    unittest.main()
