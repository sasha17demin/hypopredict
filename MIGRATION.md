# Migration Guide: hypopredict Package Structure

## Overview

This guide helps you transition to the clean, refactored structure of the hypopredict package. The package now follows a modular design with clear separation of concerns, making it easier to use, test, and maintain.

## Package Structure

The hypopredict package is organized into focused modules:

```
hypopredict/
├── __init__.py              # Package entry point
├── compressor.py            # Data loading and HG event identification
├── person.py                # Person class for patient data management
├── train_test_split.py      # Data chunking and splitting utilities
├── feature_extraction.py    # Feature extraction from sensor data
├── cv.py                    # Cross-validation splitting
└── params.py                # Dataset parameters and constants
```

## Import Patterns

### Recommended Import Style

```python
# Import specific modules
import hypopredict.compressor as comp
import hypopredict.train_test_split as tts
import hypopredict.feature_extraction as fe
from hypopredict.person import Person
from hypopredict.cv import CV_splitter
from hypopredict import params
```

### Alternative Import Style

```python
# Import specific functions/classes
from hypopredict.compressor import gdrive_to_pandas, identify_hg_events
from hypopredict.train_test_split import chunkify, chunkify_df
from hypopredict.feature_extraction import extract_features, extract_ecg_features
from hypopredict.person import Person
```

## Module Migration Guide

### 1. Data Loading (`compressor.py`)

**Purpose:** Load glucose data and identify hypoglycemic (HG) events

**Key Functions:**
- `gdrive_to_pandas(link)` - Load data from Google Drive
- `identify_hg_events(glucose_df, threshold=3.9, min_duration=15)` - Identify HG events
- `plot_hg_events(person)` - Visualize glucose levels and HG events

**Constants:**
- `GLUCOSE_ID_LINKS` - Pre-configured Google Drive links for glucose data

**Usage:**
```python
import hypopredict.compressor as comp

# Load glucose data from Google Drive
glucose_df = comp.gdrive_to_pandas(comp.GLUCOSE_ID_LINKS[0])

# Identify HG events
hg_events = comp.identify_hg_events(glucose_df, threshold=3.9, min_duration=15)
```

### 2. Patient Data Management (`person.py`)

**Purpose:** Encapsulate patient-specific data and operations

**Main Class:** `Person`

**Key Methods:**
- `__init__(ID, ecg_dir=None)` - Initialize with patient ID
- `load_HG_data(glucose_src, min_duration=15, threshold=3.9)` - Load glucose data and identify HG events
- `load_ECG_day(day, warning=True)` - Load ECG data for a specific day

**Usage:**
```python
from hypopredict.person import Person

# Initialize person
person = Person(ID=1, ecg_dir='/path/to/ecg/data')

# Load glucose data (from local or gdrive)
person.load_HG_data(glucose_src='local', min_duration=15, threshold=3.9)

# Load ECG data for day 4
person.load_ECG_day(day=4)

# Access the data
ecg_data = person.ecg[4]  # ECG data for day 4
hg_events = person.hg_events  # HG events DataFrame
```

### 3. Data Chunking (`train_test_split.py`)

**Purpose:** Split continuous sensor data into overlapping chunks for model training

**Key Functions:**
- `chunkify(person_days, chunk_size, step_size, ecg_dir)` - Chunk multiple person-days
- `chunkify_day(person_day, chunk_size, step_size, ecg_dir)` - Chunk a single day
- `chunkify_df(df, chunk_size, step_size)` - Chunk a DataFrame

**Usage:**
```python
import hypopredict.train_test_split as tts
import pandas as pd

# Define chunk parameters
chunk_size = pd.Timedelta(minutes=5)
step_size = pd.Timedelta(minutes=1)

# Chunkify multiple days
person_days = [11, 12, 13, 14]  # Person 1, days 1-4
chunks_all = tts.chunkify(
    person_days=person_days,
    chunk_size=chunk_size,
    step_size=step_size,
    ecg_dir='/path/to/ecg/data'
)

# Chunkify a single day
person_day, chunks = tts.chunkify_day(
    person_day=11,
    chunk_size=chunk_size,
    step_size=step_size,
    ecg_dir='/path/to/ecg/data'
)

# Chunkify a DataFrame directly
chunks = tts.chunkify_df(ecg_df, chunk_size=chunk_size, step_size=step_size)
```

### 4. Feature Extraction (`feature_extraction.py`)

**Purpose:** Extract statistical and ECG-specific features from chunks

**Key Functions:**
- `extract_features(chunks)` - Extract statistical features (mean, std, min, max, etc.)
- `extract_ecg_features(chunks, ecg_column='EcgWaveform', sampling_rate=250)` - Extract ECG-specific features
- `extract_hrv_features(chunks, ecg_column='EcgWaveform', sampling_rate=250)` - Extract HRV features
- `extract_combined_features_sequential(chunks, ...)` - Extract all features sequentially

**Usage:**
```python
import hypopredict.feature_extraction as fe

# Extract statistical features
stat_features = fe.extract_features(chunks)

# Extract ECG features
ecg_features = fe.extract_ecg_features(
    chunks,
    ecg_column='EcgWaveform',
    sampling_rate=250,
    verbose=True
)

# Extract HRV features
hrv_features = fe.extract_hrv_features(
    chunks,
    ecg_column='EcgWaveform',
    sampling_rate=250
)

# Extract all features at once
all_features = fe.extract_combined_features_sequential(
    chunks,
    ecg_column='EcgWaveform',
    sampling_rate=250,
    include_stats=True,
    include_ecg=True,
    include_hrv=True
)
```

### 5. Cross-Validation (`cv.py`)

**Purpose:** Create train-validation splits that respect temporal structure

**Main Class:** `CV_splitter`

**Key Methods:**
- `__init__(ecg_dir, glucose_src='local', n_splits=5, random_state=17)` - Initialize splitter
- `get_splits(days)` - Generate random splits of days
- `validate(splits, verbose=False)` - Ensure each split has HG events

**Usage:**
```python
from hypopredict.cv import CV_splitter
from hypopredict import params

# Initialize CV splitter
cv_splitter = CV_splitter(
    ecg_dir='/path/to/ecg/data',
    glucose_src='local',
    n_splits=5,
    random_state=17
)

# Generate splits from training days
splits = cv_splitter.get_splits(params.TRAIN_DAYS)

# Validate splits (check for HG events in each fold)
valid, hg_proportions = cv_splitter.validate(splits, verbose=True)
```

### 6. Dataset Parameters (`params.py`)

**Purpose:** Centralized dataset configuration and day definitions

**Key Constants:**
- `ALL_DAYS` - All available person-days
- `TRAIN_DAYS` - Days designated for training
- `TEST_DAYS` - Days designated for testing
- `DEMO_DAYS` - Days for demonstrations (high HG proportion)
- `INVALID_DAYS` - Days with incomplete/missing data
- `HG_DAYS` - Days containing HG events
- `ZERO_DAYS` - Days with no HG events

**Usage:**
```python
from hypopredict import params

# Use predefined day sets
train_days = params.TRAIN_DAYS
test_days = params.TEST_DAYS
demo_days = params.DEMO_DAYS

# Check day properties
has_hg = 13 in params.HG_DAYS
is_valid = 32 not in params.INVALID_DAYS
```

## Configuration and Environment

### Environment Variables

The package uses environment variables for paths. Create a `.env` file based on `.env.sample`:

```bash
# .env
GLUCOSE_PATH=/path/to/glucose/data
ECG_PATH=/path/to/ecg/data
```

Load environment variables in your code:

```python
import os
from dotenv import load_dotenv

load_dotenv()

glucose_path = os.getenv('GLUCOSE_PATH')
ecg_path = os.getenv('ECG_PATH')
```

## Common Workflows

### Workflow 1: Load Patient Data and Identify HG Events

```python
from hypopredict.person import Person
import os

# Initialize person
person = Person(ID=1, ecg_dir=os.getenv('ECG_PATH'))

# Load glucose data and identify HG events
person.load_HG_data(glucose_src='local', min_duration=15, threshold=3.9)

# Load ECG for a specific day
person.load_ECG_day(day=4)

# Access data
hg_events = person.hg_events
ecg_data = person.ecg[4]
```

### Workflow 2: Create Training Dataset

```python
import hypopredict.train_test_split as tts
import hypopredict.feature_extraction as fe
from hypopredict import params
import pandas as pd
import os

# Define parameters
chunk_size = pd.Timedelta(minutes=5)
step_size = pd.Timedelta(minutes=1)
ecg_dir = os.getenv('ECG_PATH')

# Chunk training days
chunks_all = tts.chunkify(
    person_days=params.TRAIN_DAYS[:5],  # First 5 training days
    chunk_size=chunk_size,
    step_size=step_size,
    ecg_dir=ecg_dir
)

# Extract features from chunks
for person_day, chunks in chunks_all.items():
    features = fe.extract_combined_features_sequential(
        chunks,
        ecg_column='EcgWaveform',
        sampling_rate=250,
        include_stats=True,
        include_ecg=True,
        include_hrv=True,
        verbose=True
    )
    print(f"Day {person_day}: {features.shape}")
```

### Workflow 3: Cross-Validation Setup

```python
from hypopredict.cv import CV_splitter
from hypopredict import params
import os

# Initialize CV splitter
cv = CV_splitter(
    ecg_dir=os.getenv('ECG_PATH'),
    glucose_src='local',
    n_splits=5,
    random_state=17
)

# Get splits
splits = cv.get_splits(params.TRAIN_DAYS)

# Validate splits
valid, hg_props = cv.validate(splits, verbose=True)

# Use splits for training
for i, split in enumerate(splits):
    print(f"Fold {i+1}: {len(split)} days, {hg_props[i]:.2%} HG")
```

## Best Practices

### 1. Always Use Environment Variables for Paths

```python
import os
from dotenv import load_dotenv

load_dotenv()

ecg_dir = os.getenv('ECG_PATH')
glucose_path = os.getenv('GLUCOSE_PATH')
```

### 2. Use Person Class for Patient Data

The `Person` class encapsulates patient data and provides a clean interface:

```python
from hypopredict.person import Person

person = Person(ID=1, ecg_dir=ecg_dir)
person.load_HG_data(glucose_src='local')
person.load_ECG_day(day=4)
```

### 3. Use Predefined Day Sets from params

```python
from hypopredict import params

# Instead of hardcoding days
train_days = params.TRAIN_DAYS
test_days = params.TEST_DAYS
```

### 4. Handle Warnings Appropriately

Some days have multiple ECG files (gaps in recording). Handle warnings:

```python
person.load_ECG_day(day=4, warning=True)  # Show warnings
# or
person.load_ECG_day(day=4, warning=False)  # Suppress warnings
```

### 5. Use Appropriate Chunk Sizes

```python
import pandas as pd

# 5-minute chunks with 1-minute overlap
chunk_size = pd.Timedelta(minutes=5)
step_size = pd.Timedelta(minutes=1)
```

## Troubleshooting

### Issue: "GLUCOSE_PATH not found"

**Solution:** Create a `.env` file with the required paths:
```bash
GLUCOSE_PATH=/path/to/glucose/data
ECG_PATH=/path/to/ecg/data
```

### Issue: "Multiple files for day warning"

**Explanation:** Some days have multiple ECG recording sessions, creating gaps in the data.

**Solution:** Be aware of potential gaps when analyzing time-series data. Check if HG events fall within recorded times.

### Issue: "No glucose for ECG times"

**Solution:** Some days (32, 33, 34, 91, 94) have incomplete data. Use `params.INVALID_DAYS` to filter them out:

```python
from hypopredict import params

valid_days = [day for day in all_days if day not in params.INVALID_DAYS]
```

### Issue: NeuroKit2 warnings during feature extraction

**Explanation:** Some chunks may have insufficient R-peaks for HRV calculation.

**Solution:** This is expected. Features will be set to NaN for problematic chunks. Handle NaN values appropriately:

```python
features = features.fillna(0)  # or dropna(), depending on use case
```

## PyTorch Dataset Creation

For deep learning models, create a PyTorch dataset:

```python
import torch
from torch.utils.data import Dataset, DataLoader

class HypoglycemiaDataset(Dataset):
    def __init__(self, chunks, labels):
        self.chunks = chunks
        self.labels = labels
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        label = self.labels[idx]
        
        # Convert to tensor
        chunk_tensor = torch.tensor(chunk.values, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        return chunk_tensor, label_tensor

# Usage
dataset = HypoglycemiaDataset(chunks, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
```

## GCS/Cloud Considerations

### Loading Data from Google Drive

```python
import hypopredict.compressor as comp

# Use predefined links
glucose_df = comp.gdrive_to_pandas(comp.GLUCOSE_ID_LINKS[person_id - 1])
```

### Training on GCS/Cloud

When training on cloud platforms:

1. **Upload data to GCS bucket:**
```bash
gsutil cp -r /local/data gs://your-bucket/hypopredict-data/
```

2. **Load data in training script:**
```python
import os

# Set environment variables for cloud paths
os.environ['GLUCOSE_PATH'] = '/gcs/your-bucket/hypopredict-data/glucose/'
os.environ['ECG_PATH'] = '/gcs/your-bucket/hypopredict-data/ecg/'
```

3. **Use the same Person class:**
```python
from hypopredict.person import Person

person = Person(ID=1, ecg_dir=os.getenv('ECG_PATH'))
person.load_HG_data(glucose_src='local')
```

## Summary

The refactored hypopredict package provides:

✅ Clear module organization with single responsibilities
✅ Consistent import patterns
✅ Person class for clean data management
✅ Flexible chunking utilities
✅ Comprehensive feature extraction
✅ Built-in cross-validation support
✅ Centralized parameter management

For more examples, see:
- `QUICK_REFERENCE.md` - Quick code snippets
- `notebooks/01_new_structure_usage.ipynb` - Comprehensive usage examples
- `TESTING_LOCALLY.md` - Local setup guide
