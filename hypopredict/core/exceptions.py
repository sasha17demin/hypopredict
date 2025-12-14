"""
Custom exceptions for hypopredict.
"""


class HypopredictError(Exception):
    """Base exception for hypopredict package."""


class InvalidPersonDayError(HypopredictError):
    """Raised when PersonDay validation fails."""


class DataLoadError(HypopredictError):
    """Raised when data loading fails."""


class FeatureExtractionError(HypopredictError):
    """Raised when feature extraction fails."""


class ModelError(HypopredictError):
    """Raised when model operations fail."""
