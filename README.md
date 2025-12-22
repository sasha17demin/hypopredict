# HypoPredict 🩺

<img src=https://i.pinimg.com/originals/6d/1e/2f/6d1e2ff4321a3656a26c2cdcee79fa8f.png width=400>

A machine learning system for predicting hypoglycemic events in Type 1 Diabetic patients using non-invasive ECG signals and advanced fusion modeling techniques.

## 📋 Table of Contents

- [Problem Overview](#problem-overview)
- [Package Structure](#package-structure)
- [Model Architecture](#model-architecture)
- [Data Handling](#data-handling)
- [FastAPI Integration](#fastapi-integration)
- [Installation](#installation)
- [Usage](#usage)
- [Development](#development)
- [Contributing](#contributing)
- [References](#references)

## 🎯 Problem Overview

Hypoglycemia (low blood glucose) is a critical condition that affects individuals with Type 1 Diabetes, potentially leading to severe complications if not detected early. Traditional monitoring methods require invasive blood sampling, which can be inconvenient and uncomfortable for patients.

**HypoPredict** addresses this challenge by:
- Leveraging **non-invasive ECG signals** to predict hypoglycemic events
- Using **physiological features** (heart rate, breathing patterns, activity levels) extracted from wearable sensors
- Providing **early warnings** to prevent dangerous glucose drops
- Enabling **continuous monitoring** without repeated blood sampling

The system analyzes time-series data from the D1NAMO dataset, which contains ECG, glucose, acceleration, and breathing measurements from Type 1 Diabetic patients.

## 📦 Package Structure

The HypoPredict package is organized into the following components:

```
hypopredict/
├── core/                    # Core data structures
│   ├── person.py           # Person class for patient data management
│   └── imports_dev.py      # Development imports
├── fusion/                  # Model fusion and ensemble
│   ├── bases.py            # Base fusion model classes
│   └── ml_preproc.py       # ML preprocessing for fusion
├── daniel_model/           # CNN+LSTM deep learning model
│   ├── lstmcnn.py         # LSTM+CNN architecture
│   ├── sequence.py        # Data sequence generators
│   └── utils.py           # Model utilities
├── chunker.py              # Time-series chunking utilities
├── chunk_preproc.py        # Chunk preprocessing
├── cv.py                   # Custom cross-validation splitter
├── labeler.py              # Hypoglycemic event labeling
├── feature_extraction.py   # Feature engineering
├── compressor.py           # Data compression utilities
└── params.py               # Configuration parameters

api/
├── fast.py                 # FastAPI application
└── utils.py                # API utilities
```

### Key Components:

- **Data Handling**: `chunker.py`, `chunk_preproc.py`, `labeler.py` - Process raw sensor data into labeled chunks
- **Feature Engineering**: `feature_extraction.py`, `new_features.py` - Extract physiological features from ECG and sensor data
- **Models**: `fusion/`, `daniel_model/` - Individual ML models and fusion ensemble
- **API**: `api/` - FastAPI server for model inference

## 🤖 Model Architecture

HypoPredict uses a **fusion modeling approach** that combines multiple machine learning algorithms to achieve robust predictions:

### Individual Models:

1. **XGBoost** - Gradient boosting for tabular features
2. **SVM (Support Vector Machine)** - Pattern recognition in feature space
3. **KNN (K-Nearest Neighbors)** - Instance-based learning
4. **CNN+LSTM** - Deep learning for sequential ECG patterns
   - Convolutional layers extract local patterns from ECG signals
   - LSTM layers capture temporal dependencies
   - Focal loss handles class imbalance
   - Architecture: LSTM → Conv1D → BatchNorm → MaxPooling → GlobalAvgPooling → Dense

### Fusion Strategy:

The fusion model combines predictions from all individual models using a meta-learner approach:
- Each base model generates predicted probabilities for hypoglycemic events
- The fusion model learns optimal weights to combine these predictions
- This ensemble approach improves robustness and generalization

### Model Features:

- **Time-series chunking**: Overlapping windows of sensor data
- **Feature extraction**: Heart rate variability, ECG morphology, breathing patterns
- **Class imbalance handling**: Focal loss and recursive undersampling
- **Temporal aggregation**: Rolling median and moving averages for smooth predictions

## 📊 Data Handling

### Cross-Validation System

HypoPredict implements a **custom cross-validation splitter** (`cv.py`) that:
- Preserves within-day temporal patterns (no random shuffling within days)
- Shuffles days within patients for robust validation
- Shuffles between patients for generalization
- Ensures each split contains at least one hypoglycemic event
- Supports configurable n-fold splits

This approach is critical for time-series medical data where temporal ordering matters.

### Recursive Undersampling

To handle severe class imbalance (hypoglycemic events are rare):
- **Undersampling** majority class (normal glucose) while preserving all minority samples
- **Recursive approach** maintains temporal coherence
- **Stratified splitting** ensures balanced representation across folds
- **Validation checks** confirm adequate representation of hypoglycemic events

### Data Pipeline:

1. **Load**: Import raw sensor data for patients and days
2. **Chunk**: Create overlapping time windows (configurable size and step)
3. **Label**: Identify hypoglycemic events and assign labels to chunks
4. **Extract**: Engineer features from raw signals
5. **Preprocess**: Scale, normalize, and prepare for model input
6. **Split**: Create train/validation/test sets with custom CV

## 🚀 FastAPI Integration

HypoPredict includes a **FastAPI server** (`api/fast.py`) for real-time predictions and system integration:

### Endpoints:

- `GET /` - Health check and welcome message
- `POST /predict_from_url` - Predict from Google Drive-hosted data
- `GET /predict_fusion_local_83` - Demo predictions for patient 83
- `GET /predict_fusion_local_64` - Demo predictions for patient 64

### Features:

- **CORS enabled** for cross-origin requests
- **Pre-loaded models** for fast inference
- **Batch predictions** with temporal aggregation
- **Multiple model support** (individual and fusion)
- **Google Drive integration** for data loading
- **Dockerized deployment** ready

### Running the API:

```bash
# Development
make run_api

# Production (Docker)
docker build -t hypopredict:latest .
docker run -p 8000:8000 hypopredict:latest
```

## 🛠️ Installation

### Prerequisites:

- Python 3.12+
- pip package manager

### Setup:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sasha17demin/hypopredict.git
   cd hypopredict
   ```

2. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Install the package (editable mode):**
   ```bash
   pip install -e .
   ```

   Or use the Makefile:
   ```bash
   make reinstall_package
   ```

4. **Verify installation:**
   ```bash
   pip freeze | grep hypopredict
   ```

## 📖 Usage

### Basic Example:

```python
from hypopredict.core.person import Person
from hypopredict.chunker import chunkify_day
import pandas as pd

# Load patient data
person = Person(patient_id=1, ecg_dir="path/to/ecg/data")
person.load_ECG_day(day=2)

# Create chunks
chunk_size = pd.Timedelta(minutes=30)
step_size = pd.Timedelta(minutes=5)
chunks = chunkify_day(
    person_day=12,
    chunk_size=chunk_size,
    step_size=step_size,
    ecg_dir="path/to/ecg/data"
)
```

### API Usage:

```python
import requests

# Predict from URL
response = requests.post(
    "http://localhost:8000/predict_from_url",
    json={"url": "https://drive.google.com/file/d/YOUR_FILE_ID/view"}
)
predictions = response.json()["predictions"]

# Get demo predictions
response = requests.get("http://localhost:8000/predict_fusion_local_83")
fusion_pred = response.json()["pred_fusion"]
cnn_pred = response.json()["pred_cnn"]
```

## 👨‍💻 Development

### Project Workflow:

1. **Update from master:**
   ```bash
   git pull origin master
   ```

2. **Create feature branch:**
   ```bash
   git checkout -b feature_name
   ```

3. **Make changes and test:**
   ```bash
   make reinstall_package  # Reload package
   make pylint             # Lint code
   make pytest             # Run tests
   ```

4. **Run CI/CD checks:**
   ```bash
   make  # Runs pylint and pytest
   ```

### Code Quality:

- **Linting**: `pylint` for code quality
- **Testing**: `pytest` for unit tests
- **Formatting**: Follow existing code style
- **Type hints**: Use where appropriate

### Docker Development:

```bash
# Build local image
make docker_build_local

# Run interactively
make docker_run_local_interactively

# Deploy to cloud
make docker_full_deploy
```

## 🤝 Contributing

We welcome contributions to HypoPredict! Here's how to get started:

1. **Fork the repository** on GitHub
2. **Create a feature branch** from `master`
3. **Make your changes** with clear commit messages
4. **Add tests** for new functionality
5. **Ensure code quality**:
   - Run `make pylint` (no critical errors)
   - Run `make pytest` (all tests pass)
6. **Submit a pull request** with a clear description

### Development Guidelines:

- Follow existing code structure and naming conventions
- Document new functions and classes with docstrings
- Add type hints where appropriate
- Keep functions focused and modular
- Write unit tests for new features
- Update README if adding major features

### Areas for Contribution:

- Model improvements and new algorithms
- Feature engineering techniques
- Data preprocessing optimizations
- API endpoint enhancements
- Documentation and examples
- Performance optimizations
- Bug fixes and testing

## 📚 References

- **D1NAMO Dataset**: [Kaggle - D1NAMO ECG Glucose Data](https://www.kaggle.com/datasets/sarabhian/d1namo-ecg-glucose-data/data)
- **Research Paper**: [Example Architecture for Hypoglycemia Prediction](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0325956)
- **Project Resources**: [Google Drive](https://drive.google.com/drive/folders/1guvUI7XiGqdeLK-qmtjcYF9KlZBd2s3E?usp=share_link)

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

## 👥 Team

**HypoPredict Team**
- Email: sasha17demin@gmail.com
- GitHub: [sasha17demin/hypopredict](https://github.com/sasha17demin/hypopredict)

---

**Note**: This project is part of ongoing research in non-invasive glucose monitoring for Type 1 Diabetic patients. The models and predictions are for research purposes and should not replace medical advice or clinical glucose monitoring devices.






