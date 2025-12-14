"""
Custom exceptions for hypopredict.
"""


class HypopredictError(Exception):
    """Base exception for hypopredict package."""
    pass


class InvalidPersonDayError(HypopredictError):
    """Raised when PersonDay validation fails."""
    pass


class DataLoadError(HypopredictError):
    """Raised when data loading fails."""
    pass


class FeatureExtractionError(HypopredictError):
    """Raised when feature extraction fails."""
    pass


class ModelError(HypopredictError):
    """Raised when model operations fail."""
    pass
