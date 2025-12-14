# Quick Reference: hypopredict Package

## Module Structure

```
hypopredict/
├── compressor.py           # Data loading & HG event detection
├── person.py               # Patient data management
├── train_test_split.py     # Data chunking utilities
├── feature_extraction.py   # Feature extraction
├── cv.py                   # Cross-validation
└── params.py              # Dataset parameters
```

## Common Import Patterns

```python
# Standard imports
import hypopredict.compressor as comp
import hypopredict.train_test_split as tts
import hypopredict.feature_extraction as fe
from hypopredict.person import Person
from hypopredict.cv import CV_splitter
from hypopredict import params

# Environment setup
import os
from dotenv import load_dotenv
load_dotenv()

# Common libraries
import pandas as pd
import numpy as np
```

## Quick Code Snippets

### 1. Load Glucose Data from Google Drive

```python
import hypopredict.compressor as comp

# Load glucose data for person 1
glucose_df = comp.gdrive_to_pandas(comp.GLUCOSE_ID_LINKS[0])

# Identify HG events (glucose < 3.9 for 15+ minutes)
hg_events = comp.identify_hg_events(glucose_df, threshold=3.9, min_duration=15)
```

### 2. Load Glucose Data Locally

```python
from hypopredict.person import Person
import os

# Initialize person
person = Person(ID=1, ecg_dir=os.getenv('ECG_PATH'))

# Load glucose data from local files
person.load_HG_data(glucose_src='local', min_duration=15, threshold=3.9)

# Access HG events
hg_events = person.hg_events
```

### 3. Load ECG Data for a Day

```python
from hypopredict.person import Person
import os

# Initialize and load
person = Person(ID=1, ecg_dir=os.getenv('ECG_PATH'))
person.load_ECG_day(day=4, warning=True)

# Access ECG data
ecg_data = person.ecg[4]
```

### 4. Chunk Data for Training

```python
import hypopredict.train_test_split as tts
import pandas as pd

# Define chunk parameters
chunk_size = pd.Timedelta(minutes=5)
step_size = pd.Timedelta(minutes=1)

# Chunk multiple person-days
chunks_all = tts.chunkify(
    person_days=[11, 12, 13],
    chunk_size=chunk_size,
    step_size=step_size,
    ecg_dir='/path/to/ecg'
)

# Access chunks for specific day
chunks_day_11 = chunks_all[11]
```

### 5. Chunk a Single Day

```python
import hypopredict.train_test_split as tts
import pandas as pd

person_day, chunks = tts.chunkify_day(
    person_day=11,
    chunk_size=pd.Timedelta(minutes=5),
    step_size=pd.Timedelta(minutes=1),
    ecg_dir='/path/to/ecg'
)
```

### 6. Chunk a DataFrame Directly

```python
import hypopredict.train_test_split as tts
import pandas as pd

chunks = tts.chunkify_df(
    df=ecg_df,
    chunk_size=pd.Timedelta(minutes=5),
    step_size=pd.Timedelta(minutes=1)
)
```

### 7. Extract Statistical Features

```python
import hypopredict.feature_extraction as fe

# Extract mean, std, min, max, quantiles, skew, kurtosis
features = fe.extract_features(chunks)
```

### 8. Extract ECG Features

```python
import hypopredict.feature_extraction as fe

# Extract ECG-specific features (R-peaks, heart rate, RR intervals)
ecg_features = fe.extract_ecg_features(
    chunks,
    ecg_column='EcgWaveform',
    sampling_rate=250,
    verbose=True
)
```

### 9. Extract HRV Features

```python
import hypopredict.feature_extraction as fe

# Extract heart rate variability features
hrv_features = fe.extract_hrv_features(
    chunks,
    ecg_column='EcgWaveform',
    sampling_rate=250,
    verbose=True
)
```

### 10. Extract All Features at Once

```python
import hypopredict.feature_extraction as fe

# Extract statistical + ECG + HRV features
all_features = fe.extract_combined_features_sequential(
    chunks,
    ecg_column='EcgWaveform',
    sampling_rate=250,
    include_stats=True,
    include_ecg=True,
    include_hrv=True,
    verbose=True
)
```

### 11. Cross-Validation Setup

```python
from hypopredict.cv import CV_splitter
from hypopredict import params

# Initialize CV splitter
cv = CV_splitter(
    ecg_dir='/path/to/ecg',
    glucose_src='local',
    n_splits=5,
    random_state=17
)

# Get splits
splits = cv.get_splits(params.TRAIN_DAYS)

# Validate splits (ensure HG events in each fold)
valid, hg_proportions = cv.validate(splits, verbose=True)
```

### 12. Use Predefined Day Sets

```python
from hypopredict import params

# Training days (20 days)
train_days = params.TRAIN_DAYS

# Test days (8 days)
test_days = params.TEST_DAYS

# Demo days (2 days with high HG proportion)
demo_days = params.DEMO_DAYS

# Invalid days (5 days with missing data)
invalid_days = params.INVALID_DAYS

# Days with HG events
hg_days = params.HG_DAYS

# Days without HG events
zero_days = params.ZERO_DAYS
```

### 13. Create PyTorch Dataset

```python
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class HypoglycemiaDataset(Dataset):
    def __init__(self, chunks, labels):
        self.chunks = chunks
        self.labels = labels
    
    def __len__(self):
        return len(self.chunks)
    
    def __getitem__(self, idx):
        # Convert chunk to tensor
        chunk = self.chunks[idx]
        chunk_array = chunk.values if hasattr(chunk, 'values') else chunk
        chunk_tensor = torch.tensor(chunk_array, dtype=torch.float32)
        
        # Convert label to tensor
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return chunk_tensor, label_tensor

# Usage
dataset = HypoglycemiaDataset(chunks, labels)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Training loop
for batch_chunks, batch_labels in dataloader:
    # Your training code here
    pass
```

### 14. Handle NaN Values in Features

```python
import pandas as pd

# Option 1: Fill with 0
features = features.fillna(0)

# Option 2: Fill with mean
features = features.fillna(features.mean())

# Option 3: Drop rows with NaN
features = features.dropna()

# Option 4: Drop columns with too many NaN
threshold = 0.5  # Drop if >50% NaN
features = features.loc[:, features.isna().mean() < threshold]
```

### 15. Complete Training Pipeline

```python
import hypopredict.train_test_split as tts
import hypopredict.feature_extraction as fe
from hypopredict import params
import pandas as pd
import os

# Parameters
chunk_size = pd.Timedelta(minutes=5)
step_size = pd.Timedelta(minutes=1)
ecg_dir = os.getenv('ECG_PATH')

# 1. Chunk data
print("Chunking data...")
chunks_all = tts.chunkify(
    person_days=params.TRAIN_DAYS[:3],
    chunk_size=chunk_size,
    step_size=step_size,
    ecg_dir=ecg_dir
)

# 2. Extract features
print("Extracting features...")
features_list = []
for person_day, chunks in chunks_all.items():
    features = fe.extract_combined_features_sequential(
        chunks,
        ecg_column='EcgWaveform',
        sampling_rate=250,
        verbose=False
    )
    features['person_day'] = person_day
    features_list.append(features)

# 3. Combine features
all_features = pd.concat(features_list, ignore_index=True)

# 4. Handle NaN
all_features = all_features.fillna(0)

print(f"Final feature shape: {all_features.shape}")
```

## Common Workflows

### Workflow 1: Single Patient Analysis

```python
from hypopredict.person import Person
import hypopredict.compressor as comp
import os

# Load person
person = Person(ID=1, ecg_dir=os.getenv('ECG_PATH'))
person.load_HG_data(glucose_src='local')
person.load_ECG_day(day=4)

# Visualize HG events
comp.plot_hg_events({"ID": person.ID, "hg_events": person.hg_events})
```

### Workflow 2: Multi-Day Feature Extraction

```python
import hypopredict.train_test_split as tts
import hypopredict.feature_extraction as fe
import pandas as pd
import os

# Chunk multiple days
chunks_all = tts.chunkify(
    person_days=[11, 12, 13, 14],
    chunk_size=pd.Timedelta(minutes=5),
    step_size=pd.Timedelta(minutes=1),
    ecg_dir=os.getenv('ECG_PATH')
)

# Extract features for each day
for person_day, chunks in chunks_all.items():
    features = fe.extract_features(chunks)
    print(f"Day {person_day}: {len(features)} feature vectors")
```

### Workflow 3: Cross-Validation Training

```python
from hypopredict.cv import CV_splitter
from hypopredict import params
import hypopredict.train_test_split as tts
import hypopredict.feature_extraction as fe
import pandas as pd
import os

# Setup CV
cv = CV_splitter(ecg_dir=os.getenv('ECG_PATH'), n_splits=5)
splits = cv.get_splits(params.TRAIN_DAYS)

# Train on each fold
for fold_idx, val_days in enumerate(splits):
    train_days = [d for d in params.TRAIN_DAYS if d not in val_days]
    
    # Chunk and extract features
    train_chunks = tts.chunkify(
        person_days=train_days,
        chunk_size=pd.Timedelta(minutes=5),
        step_size=pd.Timedelta(minutes=1),
        ecg_dir=os.getenv('ECG_PATH')
    )
    
    # Extract features
    # ... (feature extraction code)
    
    print(f"Fold {fold_idx + 1}: Train={len(train_days)}, Val={len(val_days)}")
```

## Parameter Reference

### Common Chunk Sizes

```python
import pandas as pd

# 1 minute chunks
chunk_size = pd.Timedelta(minutes=1)

# 5 minute chunks (common for HG prediction)
chunk_size = pd.Timedelta(minutes=5)

# 10 minute chunks
chunk_size = pd.Timedelta(minutes=10)

# 30 second chunks
chunk_size = pd.Timedelta(seconds=30)
```

### Common Step Sizes (Overlap)

```python
import pandas as pd

# No overlap
step_size = chunk_size

# 50% overlap
step_size = chunk_size / 2

# 1 minute step
step_size = pd.Timedelta(minutes=1)

# 30 second step
step_size = pd.Timedelta(seconds=30)
```

### ECG Sampling Rates

```python
# D1NAMO dataset default
sampling_rate = 250  # Hz

# Other common rates
sampling_rate = 128  # Hz
sampling_rate = 256  # Hz
sampling_rate = 512  # Hz
```

### HG Detection Parameters

```python
# Standard hypoglycemia threshold
threshold = 3.9  # mmol/L

# Minimum duration for HG event
min_duration = 15  # minutes

# Conservative threshold
threshold = 3.5  # mmol/L

# Relaxed duration
min_duration = 10  # minutes
```

## Environment Variables

```bash
# .env file
GLUCOSE_PATH=/path/to/glucose/data
ECG_PATH=/path/to/ecg/data
DATA_PATH=/path/to/data
MODELS_PATH=/path/to/models
```

## Error Handling Patterns

### Safe Person Loading

```python
from hypopredict.person import Person
import os

try:
    person = Person(ID=1, ecg_dir=os.getenv('ECG_PATH'))
    person.load_HG_data(glucose_src='local')
    person.load_ECG_day(day=4)
except FileNotFoundError as e:
    print(f"Data not found: {e}")
except Exception as e:
    print(f"Error loading person: {e}")
```

### Safe Feature Extraction

```python
import hypopredict.feature_extraction as fe
import warnings

# Suppress NeuroKit2 warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

try:
    features = fe.extract_ecg_features(chunks, verbose=False)
    # Handle NaN values
    features = features.fillna(0)
except Exception as e:
    print(f"Feature extraction failed: {e}")
    features = None
```

### Safe Chunk Loading

```python
import hypopredict.train_test_split as tts
from hypopredict import params

failed_days = []
successful_chunks = {}

for person_day in params.TRAIN_DAYS:
    try:
        person_day, chunks = tts.chunkify_day(
            person_day=person_day,
            chunk_size=pd.Timedelta(minutes=5),
            step_size=pd.Timedelta(minutes=1),
            ecg_dir=ecg_dir
        )
        successful_chunks[person_day] = chunks
    except Exception as e:
        print(f"Failed to chunk day {person_day}: {e}")
        failed_days.append(person_day)

print(f"Successfully chunked {len(successful_chunks)}/{len(params.TRAIN_DAYS)} days")
```

## Debugging Tips

### Check Data Availability

```python
import os
from pathlib import Path

# Check paths exist
glucose_path = os.getenv('GLUCOSE_PATH')
ecg_path = os.getenv('ECG_PATH')

print(f"Glucose path exists: {Path(glucose_path).exists()}")
print(f"ECG path exists: {Path(ecg_path).exists()}")

# List available files
if Path(ecg_path).exists():
    ecg_files = list(Path(ecg_path).glob('*.feather'))
    print(f"Found {len(ecg_files)} ECG files")
```

### Check Chunk Dimensions

```python
chunks = tts.chunkify_df(df, chunk_size=..., step_size=...)

print(f"Number of chunks: {len(chunks)}")
print(f"First chunk shape: {chunks[0].shape}")
print(f"Chunk time range: {chunks[0].index[0]} to {chunks[0].index[-1]}")
```

### Check Feature Dimensions

```python
features = fe.extract_features(chunks)

print(f"Feature matrix shape: {features.shape}")
print(f"Feature columns: {list(features.columns)}")
print(f"NaN count per column:\n{features.isna().sum()}")
```

## Performance Tips

### Parallel Processing (Future)

```python
from multiprocessing import Pool

def process_day(person_day):
    # Chunk and extract features for one day
    person_day, chunks = tts.chunkify_day(...)
    features = fe.extract_features(chunks)
    return features

# Use multiprocessing
with Pool(processes=4) as pool:
    results = pool.map(process_day, params.TRAIN_DAYS)
```

### Memory Management

```python
# Process days one at a time for large datasets
for person_day in params.TRAIN_DAYS:
    person_day, chunks = tts.chunkify_day(...)
    features = fe.extract_features(chunks)
    
    # Save features
    features.to_feather(f'features_day_{person_day}.feather')
    
    # Clear memory
    del chunks, features
```

---

For more details, see:
- `MIGRATION.md` - Comprehensive migration guide
- `TESTING_LOCALLY.md` - Setup and testing guide
- `notebooks/01_new_structure_usage.ipynb` - Full usage examples
