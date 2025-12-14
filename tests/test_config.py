"""
Tests for configuration module.
"""
import unittest
from hypopredict import config


class TestConfig(unittest.TestCase):
    """Test centralized configuration."""
    
    def test_all_days_count(self):
        """Test that ALL_DAYS has expected count."""
        self.assertEqual(len(config.ALL_DAYS), 35)
    
    def test_no_overlap_train_test(self):
        """Test that train and test sets don't overlap."""
        train_set = set(config.TRAIN_DAYS)
        test_set = set(config.TEST_DAYS)
        overlap = train_set & test_set
        self.assertEqual(len(overlap), 0, f"Train/test overlap: {overlap}")
    
    def test_no_overlap_train_demo(self):
        """Test that train and demo sets don't overlap."""
        train_set = set(config.TRAIN_DAYS)
        demo_set = set(config.DEMO_DAYS)
        overlap = train_set & demo_set
        self.assertEqual(len(overlap), 0, f"Train/demo overlap: {overlap}")
    
    def test_invalid_days_excluded(self):
        """Test that invalid days are not in train/test/demo."""
        invalid_set = set(config.INVALID_DAYS)
        train_set = set(config.TRAIN_DAYS)
        test_set = set(config.TEST_DAYS)
        demo_set = set(config.DEMO_DAYS)
        
        self.assertEqual(len(invalid_set & train_set), 0)
        self.assertEqual(len(invalid_set & test_set), 0)
        self.assertEqual(len(invalid_set & demo_set), 0)
    
    def test_hg_threshold(self):
        """Test HG threshold is reasonable."""
        self.assertEqual(config.HG_THRESHOLD, 3.9)
        self.assertGreater(config.HG_THRESHOLD, 0)
        self.assertLess(config.HG_THRESHOLD, 10)
    
    def test_hg_min_duration(self):
        """Test HG minimum duration is reasonable."""
        self.assertEqual(config.HG_MIN_DURATION, 15)
        self.assertGreater(config.HG_MIN_DURATION, 0)
    
    def test_ecg_sampling_rate(self):
        """Test ECG sampling rate is reasonable."""
        self.assertEqual(config.ECG_SAMPLING_RATE, 250)
        self.assertGreater(config.ECG_SAMPLING_RATE, 0)
    
    def test_day_format(self):
        """Test that day identifiers follow XY format."""
        for day in config.ALL_DAYS:
            person_id = day // 10
            day_num = day % 10
            self.assertGreaterEqual(person_id, 1)
            self.assertLessEqual(person_id, 9)
            self.assertGreaterEqual(day_num, 1)
            self.assertLessEqual(day_num, 6)


if __name__ == '__main__':
    unittest.main()
