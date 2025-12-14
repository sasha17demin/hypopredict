"""
Tests for feature extraction and chunking.
"""
import unittest
import pandas as pd
import numpy as np
from hypopredict.features import chunking
from hypopredict.core.data_types import PersonDay


class TestChunking(unittest.TestCase):
    """Test time-series chunking operations."""
    
    def test_chunkify_df_basic(self):
        """Test basic chunking of a DataFrame."""
        # Create mock ECG data at 250 Hz sampling rate
        sampling_rate_hz = 250
        period_ms = int(1000 / sampling_rate_hz)  # 4ms at 250 Hz
        dates = pd.date_range('2023-01-01', periods=1000, freq=f'{period_ms}ms')
        ecg_data = pd.DataFrame({
            'EcgWaveform': np.random.randn(1000)
        }, index=dates)
        
        chunk_size = pd.Timedelta(seconds=10)
        step_size = pd.Timedelta(seconds=5)
        
        chunks = chunking.chunkify_df(ecg_data, chunk_size, step_size)
        
        # Should have multiple chunks
        self.assertGreater(len(chunks), 0)
        
        # Each chunk should be a DataFrame
        for chunk in chunks:
            self.assertIsInstance(chunk, pd.DataFrame)
    
    def test_get_HG_onset_times_basic(self):
        """Test getting HG onset times."""
        # Create mock glucose data with clear onsets
        dates = pd.date_range('2023-01-01', periods=100, freq='5min')
        glucose_values = [5.0] * 20 + [3.5] * 20 + [5.0] * 20 + [3.5] * 20 + [5.0] * 20
        glucose_data = pd.DataFrame({
            'glucose': glucose_values
        }, index=dates)
        
        onset_times = chunking.get_HG_onset_times(glucose_data, threshold=3.9)
        
        # Should find onset times
        self.assertIsInstance(onset_times, list)
        # Should have at least one onset
        self.assertGreater(len(onset_times), 0)
    
    def test_get_HG_onset_times_separation(self):
        """Test that onset times are properly separated."""
        # Create data with multiple drops
        dates = pd.date_range('2023-01-01', periods=100, freq='5min')
        glucose_values = [3.0] * 100  # All below threshold
        glucose_data = pd.DataFrame({
            'glucose': glucose_values
        }, index=dates)
        
        onset_times = chunking.get_HG_onset_times(glucose_data, threshold=3.9)
        
        # Should only have one onset (all points are continuous)
        self.assertEqual(len(onset_times), 1)
    
    def test_generate_target_labels(self):
        """Test generating target labels for chunks."""
        # Create mock chunks
        dates1 = pd.date_range('2023-01-01 00:00', periods=100, freq='4ms')
        dates2 = pd.date_range('2023-01-01 00:05', periods=100, freq='4ms')
        chunks = [
            pd.DataFrame({'ecg': np.random.randn(100)}, index=dates1),
            pd.DataFrame({'ecg': np.random.randn(100)}, index=dates2)
        ]
        
        # Create onset time within forecast window of first chunk
        onset_times = [pd.Timestamp('2023-01-01 00:00:30')]
        forecast_window = pd.Timedelta(minutes=1)
        
        labels = chunking.generate_target_labels(chunks, onset_times, forecast_window)
        
        # Should return list of labels
        self.assertIsInstance(labels, list)
        self.assertEqual(len(labels), len(chunks))
        
        # Labels should be 0 or 1
        for label in labels:
            self.assertIn(label, [0, 1])
    
    def test_train_test_split_chunks(self):
        """Test splitting chunks into train/test sets."""
        # Create mock chunks
        num_chunks = 100
        chunks = [
            pd.DataFrame({'ecg': np.random.randn(50)}) 
            for _ in range(num_chunks)
        ]
        target_labels = [0] * 50 + [1] * 50
        
        X_train, X_test, y_train, y_test = chunking.train_test_split_chunks(
            chunks=chunks,
            chunk_size=50,
            step_size=25,
            target_labels=target_labels,
            forecast_window=pd.Timedelta(minutes=30),
            sampling_rate=250,
            test_size=0.2
        )
        
        # Test set should be roughly 20%
        self.assertGreater(len(X_test), 0)
        self.assertLess(len(X_test), len(chunks))
        
        # Train set should have matching labels
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_test), len(y_test))


if __name__ == '__main__':
    unittest.main()
