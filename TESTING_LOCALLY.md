# Testing Locally: Setup and Validation Guide

## Overview

This guide walks you through setting up the hypopredict package locally, running tests, and validating your installation.

## Prerequisites

- Python 3.8+ (recommended: Python 3.10 or 3.12)
- Git
- pyenv (optional but recommended for Python version management)
- pip

## Step 1: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/sasha17demin/hypopredict.git
cd hypopredict

# Check available branches
git branch -a
```

## Step 2: Checkout the PR Branch

If you're testing a specific PR branch:

```bash
# Checkout the PR branch (replace with actual branch name)
git checkout <branch-name>

# Or checkout a remote PR branch
git fetch origin pull/<PR-NUMBER>/head:<branch-name>
git checkout <branch-name>

# Verify you're on the correct branch
git branch
```

## Step 3: Set Up Python Environment

### Option A: Using pyenv (Recommended)

```bash
# Install Python 3.12 (if not already installed)
pyenv install 3.12.0

# Create virtual environment
pyenv virtualenv 3.12.0 hypopredict

# Activate the virtual environment
pyenv activate hypopredict

# Alternatively, set local Python version
pyenv local hypopredict
```

### Option B: Using venv

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### Verify Virtual Environment

```bash
# Check Python version
python --version

# Check pip version
pip --version

# Verify you're in the virtual environment
which python  # On macOS/Linux
where python  # On Windows
```

## Step 4: Install Dependencies

### Install the Package in Editable Mode

Installing in editable mode allows code changes to be reflected immediately without reinstalling:

```bash
# Upgrade pip first
pip install --upgrade pip

# Install package in editable mode
pip install -e .

# This will install all dependencies from requirements.txt
```

### Verify Installation

```bash
# Check installed packages
pip freeze | grep hypopredict

# Should show something like:
# hypopredict==0.1

# Test import
python -c "import hypopredict; print('✓ hypopredict imported successfully')"
```

### Install Development Dependencies (Optional)

```bash
# Install all development tools
pip install pytest pylint ipdb jupyterlab
```

## Step 5: Configure Environment Variables

### Create Environment File

```bash
# Copy the sample environment file
cp .env.sample .env

# Edit .env with your paths
nano .env  # or use your preferred editor
```

### Example .env Configuration

```bash
# .env
GLUCOSE_PATH=/Users/yourname/code/hypopredict/data/glucose/
ECG_PATH=/Users/yourname/code/hypopredict/data/ecg/

# Optional: Add other paths
DATA_PATH=/Users/yourname/code/hypopredict/data/
MODELS_PATH=/Users/yourname/code/hypopredict/models/
```

### Verify Environment Variables

```bash
# Source the .env file (if using direnv)
direnv allow

# Or load manually in Python
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('GLUCOSE_PATH:', os.getenv('GLUCOSE_PATH'))"
```

## Step 6: Run Tests

### Run All Tests

```bash
# Using pytest
pytest

# With verbose output
pytest -v

# With coverage report
pytest --cov=hypopredict --cov-report=html
```

### Run Specific Tests

```bash
# Run a specific test file
pytest tests/test_sample.py

# Run a specific test function
pytest tests/test_sample.py::test_function_name

# Run tests matching a pattern
pytest -k "test_person"
```

### Expected Test Output

```
============================= test session starts ==============================
platform darwin -- Python 3.12.0, pytest-9.0.2, pluggy-1.5.0
collected X items

tests/test_sample.py .....                                               [100%]

============================== X passed in 0.05s ===============================
```

## Step 7: Validate Installation with Quick Tests

### Test 1: Import All Modules

```bash
python -c "
import hypopredict.compressor as comp
import hypopredict.train_test_split as tts
import hypopredict.feature_extraction as fe
from hypopredict.person import Person
from hypopredict.cv import CV_splitter
from hypopredict import params
print('✓ All modules imported successfully')
"
```

### Test 2: Test Basic Functionality

```bash
python << 'EOF'
from hypopredict import train_test_split as tts
result = tts.hello()
print(f"✓ Function call successful: {result}")
EOF
```

### Test 3: Load Sample Data (if available)

```bash
python << 'EOF'
import os
from dotenv import load_dotenv
from hypopredict.person import Person

load_dotenv()

# Check if data paths are set
glucose_path = os.getenv('GLUCOSE_PATH')
ecg_path = os.getenv('ECG_PATH')

if glucose_path and ecg_path:
    print(f"✓ GLUCOSE_PATH: {glucose_path}")
    print(f"✓ ECG_PATH: {ecg_path}")
    
    # Try to initialize a Person (doesn't load data yet)
    person = Person(ID=1, ecg_dir=ecg_path)
    print(f"✓ Person object created: ID={person.ID}")
else:
    print("⚠ Environment variables not set. Configure .env file.")
EOF
```

## Step 8: Run Jupyter Notebooks

### Launch Jupyter Lab

```bash
# Start Jupyter Lab
jupyter lab

# Or Jupyter Notebook
jupyter notebook
```

### Open Example Notebooks

1. Navigate to `notebooks/` directory
2. Open any example notebook (e.g., `Sasha_d1_get_familiar.ipynb`)
3. Run cells to validate functionality

### Create Test Notebook

Create a new notebook to test the installation:

```python
# Cell 1: Imports
import hypopredict.compressor as comp
import hypopredict.train_test_split as tts
from hypopredict.person import Person
from hypopredict import params
import os
from dotenv import load_dotenv

# Cell 2: Load environment
load_dotenv()
print("GLUCOSE_PATH:", os.getenv('GLUCOSE_PATH'))
print("ECG_PATH:", os.getenv('ECG_PATH'))

# Cell 3: Check params
print("Training days:", params.TRAIN_DAYS)
print("Test days:", params.TEST_DAYS)

# Cell 4: Test Person class
person = Person(ID=1, ecg_dir=os.getenv('ECG_PATH'))
print(f"Person created: ID={person.ID}")
```

## Step 9: Validate Package Rebuild

If you make changes to the package:

### Reinstall Package

```bash
# Using Makefile (if available)
make reinstall_package

# Or manually
pip uninstall hypopredict -y
pip install -e .
```

### Verify Changes

```bash
# Check installation
pip show hypopredict

# Test import with changes
python -c "import hypopredict; print('Package reloaded successfully')"
```

### Restart Jupyter Kernel

If testing in Jupyter:
1. Go to `Kernel` → `Restart Kernel`
2. Re-run your cells

## Common Issues and Troubleshooting

### Issue 1: Import Errors

**Symptom:** `ModuleNotFoundError: No module named 'hypopredict'`

**Solutions:**
```bash
# Ensure you're in the correct virtual environment
which python

# Reinstall in editable mode
pip install -e .

# Check if package is installed
pip list | grep hypopredict
```

### Issue 2: Environment Variables Not Loading

**Symptom:** `None` values when accessing `os.getenv()`

**Solutions:**
```python
# Explicitly load .env file
from dotenv import load_dotenv
load_dotenv()

# Check if .env file exists
import os
print(os.path.exists('.env'))

# Print environment variables
import os
print(os.environ.get('GLUCOSE_PATH'))
```

### Issue 3: Dependencies Not Installed

**Symptom:** `ModuleNotFoundError` for dependencies (pandas, numpy, etc.)

**Solutions:**
```bash
# Install requirements explicitly
pip install -r requirements.txt

# Check for missing dependencies
pip check

# Update all packages
pip install --upgrade -r requirements.txt
```

### Issue 4: Data Path Errors

**Symptom:** `FileNotFoundError` when loading data

**Solutions:**
```bash
# Verify data directories exist
ls -la $GLUCOSE_PATH
ls -la $ECG_PATH

# Check .env configuration
cat .env

# Use absolute paths in .env
# /Users/username/... instead of ~/...
```

### Issue 5: NeuroKit2 Warnings

**Symptom:** Runtime warnings from NeuroKit2 during feature extraction

**Explanation:** Normal behavior when processing chunks with insufficient R-peaks

**Solution:** These warnings are expected and can be ignored or suppressed:
```python
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
```

### Issue 6: Jupyter Kernel Issues

**Symptom:** Kernel crashes or doesn't recognize installed package

**Solutions:**
```bash
# Install ipykernel in your virtual environment
pip install ipykernel

# Add your virtual environment to Jupyter
python -m ipykernel install --user --name=hypopredict --display-name "Python (hypopredict)"

# Restart Jupyter and select the correct kernel
```

### Issue 7: Git Branch Issues

**Symptom:** Can't checkout PR branch

**Solutions:**
```bash
# Fetch all branches
git fetch --all

# List remote branches
git branch -r

# Force checkout if needed (be careful!)
git fetch origin
git checkout -b <branch-name> origin/<branch-name>

# Or reset to remote state
git reset --hard origin/<branch-name>
```

## Running the Full Test Suite

### Using Make (if available)

```bash
# Run tests
make test

# Run tests with coverage
make test-coverage

# Lint code
make lint

# Clean build artifacts
make clean
```

### Manual Test Execution

```bash
# Run all tests with coverage
pytest --cov=hypopredict --cov-report=term-missing --cov-report=html

# Open coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## Continuous Testing During Development

### Use pytest-watch for Auto-Testing

```bash
# Install pytest-watch
pip install pytest-watch

# Run tests automatically on file changes
ptw

# Or with specific options
ptw -- --cov=hypopredict -v
```

### Use Jupyter Auto-Reload

In Jupyter notebooks, use auto-reload for development:

```python
# At the top of your notebook
%load_ext autoreload
%autoreload 2

# Now your changes will be reflected automatically
import hypopredict.compressor as comp
```

## Validation Checklist

Before considering your setup complete, verify:

- [ ] Python version is 3.8+ (recommended 3.10 or 3.12)
- [ ] Virtual environment is activated
- [ ] hypopredict package is installed in editable mode (`pip install -e .`)
- [ ] All dependencies are installed (`pip list`)
- [ ] Environment variables are configured (`.env` file exists)
- [ ] All modules can be imported without errors
- [ ] Tests pass (`pytest`)
- [ ] Jupyter Lab/Notebook launches successfully
- [ ] Can create Person objects
- [ ] Sample notebooks run without errors (if data available)

## Next Steps

After successful local setup:

1. **Read the Migration Guide:** `MIGRATION.md` for package structure details
2. **Check Quick Reference:** `QUICK_REFERENCE.md` for common code patterns
3. **Explore Examples:** `notebooks/01_new_structure_usage.ipynb` for comprehensive examples
4. **Start Developing:** Make changes and run tests frequently

## Getting Help

If you encounter issues:

1. Check this troubleshooting guide
2. Review error messages carefully
3. Check GitHub Issues for similar problems
4. Reach out to the team via:
   - GitHub Issues
   - Team Slack/Discord
   - Email: sasha17demin@gmail.com

## Useful Commands Reference

```bash
# Environment Management
pyenv activate hypopredict          # Activate pyenv environment
pyenv deactivate                    # Deactivate pyenv environment
source venv/bin/activate            # Activate venv

# Package Management
pip install -e .                    # Install in editable mode
pip uninstall hypopredict           # Uninstall package
pip list                            # List installed packages
pip freeze > requirements.txt       # Save dependencies

# Testing
pytest                              # Run all tests
pytest -v                           # Verbose output
pytest -k "test_name"               # Run specific tests
pytest --cov=hypopredict            # With coverage

# Git
git status                          # Check status
git branch                          # List branches
git checkout <branch>               # Switch branch
git pull origin <branch>            # Pull latest changes

# Jupyter
jupyter lab                         # Launch Jupyter Lab
jupyter notebook                    # Launch Jupyter Notebook
jupyter kernelspec list             # List available kernels
```

---

**Happy Testing! 🚀**
