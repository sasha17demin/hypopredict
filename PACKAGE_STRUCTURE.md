# HypoPredict Package Structure (v0.2.0)

## Overview

This document describes the restructured HypoPredict package, which now has a clear separation of concerns and improved maintainability.

## Package Structure

```
hypopredict/
├── core/               # Core data types and domain objects
│   ├── __init__.py
│   ├── data_types.py   # Type-safe PersonDay dataclass
│   ├── exceptions.py   # Custom exceptions
│   └── person.py       # Person class for managing per-person data
│
├── data/               # Data loading and labeling
│   ├── __init__.py
│   ├── loaders.py      # CSV/feather loading, Google Drive integration
│   └── labels.py       # Hypoglycemia event identification
│
├── features/           # Feature extraction and engineering
│   ├── __init__.py
│   └── chunking.py     # Time-series chunking (sliding windows)
│
├── datasets/           # PyTorch dataset implementations
│   ├── __init__.py
│   └── panel_dataset.py  # PanelDataset for time-series
│
├── modeling/           # Neural network architectures
│   ├── __init__.py
│   └── fusion.py       # FusionNN for multimodal prediction
│
├── api_utils/          # API utilities
│   ├── __init__.py
│   └── inference.py    # Inference pipeline for FastAPI
│
├── config.py           # Centralized configuration
├── cv.py               # Cross-validation utilities
├── feature_extraction.py  # Statistical feature extraction
├── params.py           # Legacy parameters (deprecated)
│
# Backward compatibility shims (deprecated)
├── person.py           # Redirects to core.person
├── compressor.py       # Redirects to data.*
└── train_test_split.py # Redirects to features.chunking
```

## Key Changes

### 1. Type-Safe PersonDay Class

Replaces fragile integer encoding (e.g., `35 = person 3, day 5`):

```python
# Old approach (error-prone)
person_day = 35
person_id = person_day // 10  # 3
day = person_day % 10          # 5

# New approach (type-safe)
from hypopredict.core import PersonDay

person_day = PersonDay(person_id=3, day=5)
print(person_day)  # "Person3_Day5"

# Backward compatibility
person_day = PersonDay.from_legacy_id(35)
legacy_id = person_day.to_legacy_id()  # 35
```

### 2. Separated Data Operations

**Before:** `compressor.py` handled both data loading AND label generation

**After:** Split into focused modules:
- `data/loaders.py`: CSV loading, Google Drive integration
- `data/labels.py`: HG event identification

```python
# New imports
from hypopredict.data import loaders, labels

# Load glucose data
glucose_df = loaders.gdrive_to_pandas(link)

# Identify HG events
hg_events = labels.identify_hg_events(glucose_df, threshold=3.9, min_duration=15)
```

### 3. Honest Naming: train_test_split → chunking

**Before:** `train_test_split.py` actually chunks data (misleading name)

**After:** `features/chunking.py` clearly describes what it does

```python
# New imports
from hypopredict.features import chunking

# Chunk data with sliding windows
chunks = chunking.chunkify_df(
    df,
    chunk_size=pd.Timedelta(seconds=60),
    step_size=pd.Timedelta(seconds=30)
)
```

### 4. Centralized Configuration

All constants and parameters now in `config.py`:

```python
from hypopredict import config

# Day lists
train_days = config.TRAIN_DAYS
test_days = config.TEST_DAYS
invalid_days = config.INVALID_DAYS

# Parameters
threshold = config.HG_THRESHOLD  # 3.9 mmol/L
min_duration = config.HG_MIN_DURATION  # 15 minutes
sampling_rate = config.ECG_SAMPLING_RATE  # 250 Hz
```

### 5. PyTorch Integration

New `PanelDataset` for clean PyTorch integration:

```python
from hypopredict.datasets import PanelDataset
from torch.utils.data import DataLoader

# Create dataset
dataset = PanelDataset(chunks, labels, person_days)

# Use with DataLoader
loader = DataLoader(
    dataset,
    batch_size=32,
    collate_fn=PanelDataset.collate_fn
)
```

### 6. Fusion Neural Network

Skeleton implementation for multimodal prediction:

```python
from hypopredict.modeling import FusionNN

# Create model
model = FusionNN(
    ecg_input_dim=1,
    hidden_dim=128,
    num_classes=2,
    use_attention=True
)

# Make predictions
predictions = model.predict(ecg_data)
probabilities = model.predict_proba(ecg_data)
```

### 7. API Inference Bridge

Clean interface for FastAPI integration:

```python
from hypopredict.api_utils import InferencePipeline

# Load model
pipeline = InferencePipeline(model_path='model.keras')

# Run inference
predictions = pipeline.predict(data)
probabilities = pipeline.predict_proba(data)
```

## Migration Guide

### For Existing Code

Old imports will continue to work with deprecation warnings:

```python
# Old import (deprecated but functional)
from hypopredict.person import Person
# DeprecationWarning: Use 'from hypopredict.core import Person'

from hypopredict.compressor import identify_hg_events
# DeprecationWarning: Use 'from hypopredict.data.labels import identify_hg_events'

from hypopredict.train_test_split import chunkify
# DeprecationWarning: Use 'from hypopredict.features.chunking import chunkify'
```

### Recommended Updates

1. **Update imports** to use new structure:
   ```python
   from hypopredict.core import Person, PersonDay
   from hypopredict.data import loaders, labels
   from hypopredict.features import chunking
   from hypopredict import config
   ```

2. **Use PersonDay** for type safety:
   ```python
   # Instead of: person_day = 35
   person_day = PersonDay(3, 5)
   ```

3. **Use config module** for constants:
   ```python
   # Instead of: from hypopredict.params import ALL_DAYS
   from hypopredict import config
   train_days = config.TRAIN_DAYS
   ```

## Benefits

### Single Responsibility
- Each module has one clear purpose
- Easier to understand and maintain

### Discoverability
- Logical grouping makes it easy to find functionality
- Clear module names (`data`, `features`, `modeling`)

### Type Safety
- PersonDay eliminates encoding errors
- Explicit types catch bugs early

### Maintainability
- Clear separation of concerns
- Easy to add new features to the right place

### API Integration
- Clean bridge between FastAPI and ML pipeline
- Reusable inference utilities

## Testing

Run the full test suite:

```bash
# All tests
pytest tests/

# Specific modules
pytest tests/test_core.py
pytest tests/test_data.py
pytest tests/test_features.py
pytest tests/test_config.py
```

**Test Coverage:**
- 30 passing tests
- 13 skipped (PyTorch optional)
- Core functionality fully tested

## Development

### Linting

```bash
# Lint all code
make pylint

# Or specific modules
pylint hypopredict/core/*.py
pylint hypopredict/data/*.py
```

### Running Tests

```bash
# Run all tests
make pytest

# Or use pytest directly
pytest tests/ -v
```

## Future Enhancements

Potential future improvements:
1. Add more PyTorch models to `modeling/`
2. Expand `api_utils/` with more inference tools
3. Add data augmentation to `features/`
4. Create preprocessing pipelines in `data/`
5. Add experiment tracking utilities

## Questions?

For questions or issues with the new structure:
1. Check this documentation
2. Review test files for usage examples
3. Open an issue on GitHub
