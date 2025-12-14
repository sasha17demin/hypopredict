"""
Tests for core data types and structures.
"""
import unittest
from hypopredict.core.data_types import PersonDay
from hypopredict.core.exceptions import InvalidPersonDayError, HypopredictError


class TestPersonDay(unittest.TestCase):
    """Test PersonDay dataclass."""
    
    def test_valid_person_day(self):
        """Test creating valid PersonDay."""
        pd = PersonDay(3, 5)
        self.assertEqual(pd.person_id, 3)
        self.assertEqual(pd.day, 5)
    
    def test_invalid_person_id(self):
        """Test that invalid person_id raises ValueError."""
        with self.assertRaises(ValueError):
            PersonDay(0, 3)
        with self.assertRaises(ValueError):
            PersonDay(10, 3)
    
    def test_invalid_day(self):
        """Test that invalid day raises ValueError."""
        with self.assertRaises(ValueError):
            PersonDay(3, 0)
        with self.assertRaises(ValueError):
            PersonDay(3, 7)
    
    def test_from_legacy_id(self):
        """Test creating PersonDay from legacy integer encoding."""
        pd = PersonDay.from_legacy_id(35)
        self.assertEqual(pd.person_id, 3)
        self.assertEqual(pd.day, 5)
        
        pd2 = PersonDay.from_legacy_id(91)
        self.assertEqual(pd2.person_id, 9)
        self.assertEqual(pd2.day, 1)
    
    def test_to_legacy_id(self):
        """Test converting PersonDay to legacy integer encoding."""
        pd = PersonDay(3, 5)
        self.assertEqual(pd.to_legacy_id(), 35)
        
        pd2 = PersonDay(9, 1)
        self.assertEqual(pd2.to_legacy_id(), 91)
    
    def test_round_trip_conversion(self):
        """Test that legacy conversion round-trips correctly."""
        original = 35
        pd = PersonDay.from_legacy_id(original)
        converted = pd.to_legacy_id()
        self.assertEqual(original, converted)
    
    def test_str_representation(self):
        """Test string representation."""
        pd = PersonDay(3, 5)
        self.assertEqual(str(pd), "Person3_Day5")
    
    def test_immutability(self):
        """Test that PersonDay is immutable (frozen)."""
        pd = PersonDay(3, 5)
        with self.assertRaises(AttributeError):
            pd.person_id = 4


class TestExceptions(unittest.TestCase):
    """Test custom exceptions."""
    
    def test_hypopredict_error(self):
        """Test base exception."""
        with self.assertRaises(HypopredictError):
            raise HypopredictError("Test error")
    
    def test_invalid_person_day_error(self):
        """Test PersonDay-specific exception."""
        with self.assertRaises(InvalidPersonDayError):
            raise InvalidPersonDayError("Invalid day")


if __name__ == '__main__':
    unittest.main()
