"""
Centralized configuration for hypopredict.

Contains all hardcoded constants, day lists, and configuration parameters
that were previously scattered across params.py and other modules.
"""
from typing import List


# =============================================================================
# Person-Day Identifiers
# =============================================================================
# Day identifiers use format: XY where X=person_id, Y=day_number
# E.g., 35 = Person 3, Day 5

ALL_DAYS: List[int] = [
    11,  # 0% HG
    12,  # 0% HG
    13,  # 28.76% HG
    14,  # 24.42% HG
    21,  # 50.00% HG
    22,  # 22.00% HG
    23,  # 17.47% HG
    24,  # 12.03% HG
    31,  # 0% HG
    32,  # no glucose for ECG times
    33,  # no glucose for ECG times
    34,  # no glucose for ECG times
    41,  # 17.99% HG
    42,  # 0% HG
    43,  # 10.42% HG
    44,  # 0% HG
    51,  # 0% HG
    52,  # 0% HG
    53,  # 0% HG
    54,  # 0% HG
    61,  # 0% HG
    62,  # 0% HG
    63,  # 0% HG
    64,  # 7.30% HG
    71,  # 0% HG
    72,  # 0% HG
    73,  # 0% HG
    74,  # 0% HG
    81,  # 19.02% HG
    82,  # 2.42% HG
    83,  # 14.89% HG
    91,  # no glucose for ECG times
    92,  # 90.48% HG
    93,  # 5.77% HG
    94   # no glucose for ECG times
]

# Days with no glucose measures during ECG recording times
INVALID_DAYS: List[int] = [32, 33, 34, 91, 94]

# Days with at least one hypoglycemic event
HG_DAYS: List[int] = [13, 14, 21, 22, 23, 24, 41, 43, 64, 81, 82, 83, 92, 93]

# Days with no hypoglycemic events
ZERO_DAYS: List[int] = [11, 12, 31, 42, 44, 51, 52, 53, 54, 61, 62, 63, 71, 72, 73, 74]

# Demo days for testing (11% HG overall)
DEMO_DAYS: List[int] = [
    83,  # potentially starts with HG and enough variation
    64   # really imbalanced but has some HG events
]

# Test set: different people, one day per person, overall unbalanced (6.8% HG)
TEST_DAYS: List[int] = [12, 23, 31, 41, 53, 62, 73, 81]

# Training days (12% HG overall)
TRAIN_DAYS: List[int] = [
    day for day in ALL_DAYS 
    if day not in TEST_DAYS and day not in DEMO_DAYS and day not in INVALID_DAYS
]


# =============================================================================
# Hypoglycemia Detection Parameters
# =============================================================================
HG_THRESHOLD: float = 3.9  # mmol/L - glucose threshold for hypoglycemia
HG_MIN_DURATION: int = 15  # minutes - minimum duration to consider as HG event


# =============================================================================
# ECG/Sensor Parameters
# =============================================================================
ECG_SAMPLING_RATE: int = 250  # Hz
ECG_COLUMN_NAME: str = 'EcgWaveform'


# =============================================================================
# Feature Extraction Parameters
# =============================================================================
# Default chunk size for ECG segmentation
DEFAULT_CHUNK_SIZE_SECONDS: int = 60  # seconds

# Default step size for sliding window
DEFAULT_STEP_SIZE_SECONDS: int = 30  # seconds

# Forecast window for prediction (how far ahead we predict HG events)
DEFAULT_FORECAST_WINDOW_MINUTES: int = 30  # minutes


# =============================================================================
# Train/Test Split Parameters
# =============================================================================
DEFAULT_TEST_SIZE: float = 0.2  # proportion of data for testing


# =============================================================================
# Cross-Validation Parameters
# =============================================================================
DEFAULT_N_SPLITS: int = 5  # number of CV folds
DEFAULT_RANDOM_STATE: int = 17  # random seed for reproducibility


# =============================================================================
# Model Parameters
# =============================================================================
# TODO: Add model-specific parameters here as needed
# E.g., neural network architecture parameters, hyperparameters, etc.


# =============================================================================
# Data Paths (can be overridden by environment variables)
# =============================================================================
# These are loaded from environment variables in actual usage
# See Person.load_HG_data() and related functions
