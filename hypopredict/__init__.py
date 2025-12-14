"""
HypoPredict: Hypoglycemia Prediction from Multimodal Sensor Data

Restructured package with clear separation of concerns:
- core: Core data types (PersonDay, Person) and exceptions
- data: Data loading (loaders) and labeling (HG event identification)
- features: Feature extraction and chunking (sliding windows)
- datasets: PyTorch datasets for model training
- modeling: Neural network architectures (FusionNN)
- api_utils: Utilities for API inference
- config: Centralized configuration
"""

# Core functionality
from hypopredict.core import PersonDay, Person, HypopredictError

# Data operations
from hypopredict.data import (
    gdrive_to_pandas,
    identify_hg_events,
    GLUCOSE_ID_LINKS
)

# Feature extraction
from hypopredict.features import (
    chunkify,
    chunkify_day,
    chunkify_df,
    generate_target_labels
)

# Configuration
from hypopredict import config

# Version
__version__ = "0.2.0"

__all__ = [
    # Core
    'PersonDay',
    'Person',
    'HypopredictError',
    # Data
    'gdrive_to_pandas',
    'identify_hg_events',
    'GLUCOSE_ID_LINKS',
    # Features
    'chunkify',
    'chunkify_day',
    'chunkify_df',
    'generate_target_labels',
    # Config
    'config',
]
