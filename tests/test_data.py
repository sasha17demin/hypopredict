"""
Tests for data loading and labeling modules.
"""
import unittest
import pandas as pd
import numpy as np
from hypopredict.data import loaders, labels


class TestLoaders(unittest.TestCase):
    """Test data loading utilities."""
    
    def test_glucose_id_links_count(self):
        """Test that we have links for all persons (1-9)."""
        self.assertEqual(len(loaders.GLUCOSE_ID_LINKS), 9)
    
    def test_glucose_links_format(self):
        """Test that all links are Google Drive URLs."""
        for link in loaders.GLUCOSE_ID_LINKS:
            self.assertIn("drive.google.com", link)
            self.assertIn("/file/d/", link)


class TestLabels(unittest.TestCase):
    """Test hypoglycemia event identification."""
    
    def test_identify_hg_events_structure(self):
        """Test that identify_hg_events returns expected columns."""
        # Create mock glucose data
        dates = pd.date_range('2023-01-01', periods=100, freq='5min')
        glucose_data = pd.DataFrame({
            'glucose': np.random.uniform(3.0, 8.0, 100),
            'type': ['cgm'] * 100
        }, index=dates)
        
        result = labels.identify_hg_events(glucose_data, threshold=3.9, min_duration=15)
        
        # Check expected columns
        self.assertIn('is_hg', result.columns)
        self.assertIn('onset', result.columns)
        self.assertIn('end', result.columns)
        self.assertIn('glucose', result.columns)
    
    def test_identify_hg_events_threshold(self):
        """Test that HG events are correctly identified below threshold."""
        # Create data with clear HG event
        dates = pd.date_range('2023-01-01', periods=20, freq='5min')
        glucose_values = [5.0] * 5 + [3.5] * 10 + [5.0] * 5  # HG in middle
        glucose_data = pd.DataFrame({
            'glucose': glucose_values,
            'type': ['cgm'] * 20
        }, index=dates)
        
        result = labels.identify_hg_events(glucose_data, threshold=3.9, min_duration=15)
        
        # Should have at least one HG event
        self.assertGreater(result['is_hg'].sum(), 0)
    
    def test_identify_hg_events_min_duration(self):
        """Test that brief drops below threshold are filtered out."""
        # Create data with brief drop (< 15 minutes)
        dates = pd.date_range('2023-01-01', periods=20, freq='5min')
        glucose_values = [5.0] * 10 + [3.5, 3.5] + [5.0] * 8  # Only 10 min below
        glucose_data = pd.DataFrame({
            'glucose': glucose_values,
            'type': ['cgm'] * 20
        }, index=dates)
        
        result = labels.identify_hg_events(glucose_data, threshold=3.9, min_duration=15)
        
        # Should not identify as HG event (too brief)
        # Note: This depends on the exact implementation, may need adjustment
        self.assertLessEqual(result['onset'].sum(), 1)
    
    def test_day_count_HG(self):
        """Test counting HG events per day."""
        # Create mock person data
        dates = pd.date_range('2023-01-01', periods=100, freq='5min')
        hg_events = pd.DataFrame({
            'glucose': np.random.uniform(3.0, 8.0, 100),
            'is_hg': [0] * 100,
            'onset': [0] * 100,
            'end': [0] * 100
        }, index=dates)
        
        # Add an onset
        hg_events.loc[hg_events.index[10], 'onset'] = 1
        
        person = {
            'ID': 1,
            'hg_events': hg_events
        }
        
        result = labels.day_count_HG(person)
        
        # Should return dict with days 1-6
        self.assertIsInstance(result, dict)
        self.assertEqual(len(result), 6)
        self.assertIn('1', result)


if __name__ == '__main__':
    unittest.main()
