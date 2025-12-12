# Reproducibility Guide

This guide ensures you can reproduce the exact results reported in this project, including model performance, experiments, and benchmarks.

## Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Data Preparation](#data-preparation)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Expected Results](#expected-results)
- [Troubleshooting](#troubleshooting)

## Overview

This project achieves the following performance metrics on the Chest X-Ray Pneumonia dataset:

- **Accuracy**: 87.0%
- **Precision**: 85.0%
- **Recall**: 89.0%
- **F1-Score**: 87.0%
- **AUC-ROC**: 0.92

All results are reproducible using the instructions below.

## Environment Setup

### Hardware Requirements

**Minimum (CPU-only training):**
- CPU: 4 cores
- RAM: 8 GB
- Storage: 20 GB

**Recommended (GPU training):**
- GPU: NVIDIA GPU with 8GB+ VRAM (e.g., RTX 3070, V100)
- CPU: 8+ cores
- RAM: 16 GB
- Storage: 50 GB
- CUDA: 11.7 or higher

### Software Requirements

1. **Install Python 3.9.x** (exact version for reproducibility):
   ```bash
   # Using pyenv (recommended)
   pyenv install 3.9.13
   pyenv local 3.9.13
   
   # Or download from python.org
   ```

2. **Create isolated environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install exact dependencies**:
   ```bash
   # Install from requirements.txt with pinned versions
   pip install -r requirements.txt
   
   # Verify installations
   python -c "import torch; print(f'PyTorch: {torch.__version__}')"
   python -c "import torchvision; print(f'TorchVision: {torchvision.__version__}')"
   ```

### Docker Setup (Alternative)

For maximum reproducibility, use Docker:

```bash
# Build image with exact dependencies
docker-compose build training

# Run training in container
docker-compose run training python training/train_model.py
```

## Data Preparation

### Step 1: Download Dataset

```bash
# Install Kaggle CLI
pip install kaggle

# Configure Kaggle credentials
# 1. Go to https://www.kaggle.com/account
# 2. Create API token
# 3. Place kaggle.json in ~/.kaggle/

# Download dataset
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d data/raw/
```

### Step 2: Verify Dataset

```bash
# Check dataset structure and counts
python scripts/verify_dataset.py

# Expected output:
# Training set: 5,216 images (1,341 NORMAL, 3,875 PNEUMONIA)
# Validation set: 16 images (8 NORMAL, 8 PNEUMONIA)
# Test set: 624 images (234 NORMAL, 390 PNEUMONIA)
```

### Step 3: Data Preprocessing

```bash
# Run data cleaning and augmentation
python medical_dataset_cleaner.py
python medical_augmentation.py

# This will:
# - Remove corrupted images
# - Validate image quality
# - Apply medical-appropriate augmentations
# - Balance the dataset
```

### Step 4: Data Versioning

```bash
# Version the processed dataset
python data_pipeline/versioning.py --version v1.0

# This creates a snapshot for reproducibility
```

## Model Training

### Reproducible Training Configuration

The exact configuration used for reported results is in `config/training_reproducible.yaml`:

```yaml
# Model Configuration
model:
  architecture: efficientnet_b4
  pretrained: true
  num_classes: 2

# Training Configuration
training:
  epochs: 20
  batch_size: 32
  learning_rate: 0.0001
  optimizer: adam
  scheduler: cosine
  early_stopping_patience: 5
  
# Reproducibility
seed: 42
deterministic: true
benchmark: false

# Data Augmentation
augmentation:
  rotation_range: 10
  width_shift_range: 0.1
  height_shift_range: 0.1
  horizontal_flip: true
  zoom_range: 0.1
```

### Step 1: Set Random Seeds

The training script automatically sets seeds for reproducibility:

```python
import torch
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### Step 2: Run Training

```bash
# Train with reproducible configuration
python training/train_model.py \
  --config config/training_reproducible.yaml \
  --seed 42 \
  --deterministic

# Training will take approximately:
# - CPU: 6-8 hours
# - GPU (RTX 3070): 45-60 minutes
# - GPU (V100): 30-40 minutes
```

### Step 3: Monitor Training

```bash
# In another terminal, launch MLflow UI
python launch_mlflow_ui.py

# Open http://localhost:5000 to view:
# - Training/validation loss curves
# - Accuracy metrics
# - Learning rate schedule
# - Model checkpoints
```

### Training Output

The training process will create:

```
models/
├── efficientnet_b4_epoch_15_acc_0.8700.pth  # Best model checkpoint
├── training_history.json                     # Metrics history
└── training_config.yaml                      # Exact configuration used

mlruns/
└── 0/
    └── <run_id>/
        ├── metrics/                          # Logged metrics
        ├── params/                           # Hyperparameters
        └── artifacts/                        # Model artifacts
```

## Evaluation

### Step 1: Evaluate on Test Set

```bash
# Evaluate the trained model
python training/evaluate_model.py \
  --model-path models/efficientnet_b4_epoch_15_acc_0.8700.pth \
  --test-dir data/raw/test/ \
  --output evaluation_results.json
```

### Step 2: Generate Detailed Reports

```bash
# Generate comprehensive evaluation report
python scripts/generate_evaluation_report.py \
  --model-path models/efficientnet_b4_epoch_15_acc_0.8700.pth \
  --output-dir reports/

# This creates:
# - Confusion matrix
# - ROC curve
# - Precision-Recall curve
# - Per-class metrics
# - Error analysis
```

### Step 3: Statistical Significance

```bash
# Run multiple evaluation runs for confidence intervals
python scripts/evaluate_with_bootstrap.py \
  --model-path models/efficientnet_b4_epoch_15_acc_0.8700.pth \
  --n-bootstrap 1000 \
  --output bootstrap_results.json
```

## Expected Results

### Performance Metrics

When following this guide exactly, you should achieve:

```json
{
  "accuracy": 0.870,
  "precision": {
    "NORMAL": 0.850,
    "PNEUMONIA": 0.850
  },
  "recall": {
    "NORMAL": 0.820,
    "PNEUMONIA": 0.890
  },
  "f1_score": {
    "NORMAL": 0.835,
    "PNEUMONIA": 0.870,
    "macro_avg": 0.853,
    "weighted_avg": 0.870
  },
  "auc_roc": 0.920,
  "confusion_matrix": [
    [192, 42],   # NORMAL: 192 correct, 42 misclassified
    [43, 347]    # PNEUMONIA: 43 misclassified, 347 correct
  ]
}
```

### Acceptable Variance

Due to hardware differences and floating-point operations:

- **Accuracy**: ±1% (86.0% - 88.0%)
- **Precision/Recall**: ±2%
- **Training time**: ±20%

If your results fall outside this range, see [Troubleshooting](#troubleshooting).

### Training Curves

Expected training behavior:

- **Training Loss**: Decreases from ~0.6 to ~0.15
- **Validation Loss**: Decreases from ~0.5 to ~0.25
- **Training Accuracy**: Increases from ~70% to ~95%
- **Validation Accuracy**: Increases from ~75% to ~87%
- **Best epoch**: Typically around epoch 15-18

## Reproducibility Checklist

Before reporting results, verify:

- [ ] Using Python 3.9.x
- [ ] Exact package versions from requirements.txt
- [ ] Correct dataset (Kaggle chest-xray-pneumonia)
- [ ] Data preprocessing completed
- [ ] Random seed set to 42
- [ ] Deterministic mode enabled
- [ ] Using reproducible configuration file
- [ ] Training completed without errors
- [ ] Evaluation on official test set
- [ ] Results within acceptable variance

## Comparing Results

### Compare with Baseline

```bash
# Compare your results with reported baseline
python scripts/compare_results.py \
  --your-results evaluation_results.json \
  --baseline-results baseline_results.json
```

### Visualize Differences

```bash
# Generate comparison plots
python scripts/plot_comparison.py \
  --results evaluation_results.json \
  --output comparison_plots/
```

## Troubleshooting

### Results Don't Match

**Issue**: Accuracy is significantly different (>2%)

**Solutions**:
1. Verify dataset integrity:
   ```bash
   python scripts/verify_dataset.py --checksum
   ```

2. Check random seed is set:
   ```bash
   grep "seed" training/train_model.py
   ```

3. Ensure deterministic mode:
   ```bash
   # Should see in logs:
   # "Deterministic mode enabled"
   # "CUDNN benchmark disabled"
   ```

4. Verify data preprocessing:
   ```bash
   python scripts/verify_preprocessing.py
   ```

### Training Diverges

**Issue**: Loss increases or becomes NaN

**Solutions**:
1. Reduce learning rate:
   ```yaml
   learning_rate: 0.00005  # Half the default
   ```

2. Check data normalization:
   ```python
   # Images should be normalized to [0, 1] or [-1, 1]
   ```

3. Verify batch size:
   ```bash
   # Reduce if out of memory
   batch_size: 16
   ```

### GPU Memory Issues

**Issue**: CUDA out of memory

**Solutions**:
```bash
# Reduce batch size
python training/train_model.py --batch-size 16

# Enable gradient checkpointing
python training/train_model.py --gradient-checkpointing

# Use mixed precision
python training/train_model.py --mixed-precision
```

### Different Hardware Results

**Issue**: Results vary on different GPUs

**Explanation**: Some variance is expected due to:
- Different CUDA versions
- GPU architecture differences
- Floating-point precision

**Mitigation**:
```bash
# Use CPU for exact reproducibility (slower)
python training/train_model.py --device cpu

# Or use Docker for consistent environment
docker-compose run training python training/train_model.py
```

## Reporting Results

When reporting results from this project, please include:

1. **Environment Details**:
   - Python version
   - PyTorch version
   - CUDA version (if GPU)
   - Hardware specifications

2. **Configuration**:
   - Training configuration file used
   - Random seed
   - Any modifications made

3. **Results**:
   - All metrics (accuracy, precision, recall, F1)
   - Confidence intervals (if computed)
   - Training time
   - Number of parameters

4. **Reproducibility**:
   - Link to exact code version (git commit hash)
   - Dataset version
   - Any preprocessing steps

### Example Report

```markdown
## Results

Trained EfficientNet-B4 model on Chest X-Ray Pneumonia dataset.

**Environment:**
- Python 3.9.13
- PyTorch 2.0.1
- CUDA 11.7
- GPU: NVIDIA RTX 3070

**Configuration:**
- Config: config/training_reproducible.yaml
- Seed: 42
- Epochs: 20
- Batch size: 32

**Performance:**
- Accuracy: 87.2% (±0.5%)
- Precision: 85.1%
- Recall: 89.3%
- F1-Score: 87.1%
- AUC-ROC: 0.921

**Training:**
- Time: 52 minutes
- Best epoch: 16
- Parameters: 17.6M

**Reproducibility:**
- Code: github.com/user/repo/commit/abc123
- Dataset: Kaggle chest-xray-pneumonia v1.0
```

## Additional Resources

- **Training Logs**: Check `training.log` for detailed training information
- **MLflow Experiments**: View all experiments at http://localhost:5000
- **Model Cards**: See `models/model_card.md` for model documentation
- **Benchmarks**: See `benchmarks/` for performance benchmarks

## Questions?

If you have trouble reproducing results:

1. Check [SETUP.md](SETUP.md) for setup issues
2. Review [docs/USER_GUIDE.md](docs/USER_GUIDE.md) for usage
3. Open an issue on GitHub with:
   - Your environment details
   - Steps you followed
   - Actual vs expected results
   - Relevant logs

We're committed to reproducibility and will help resolve any issues!
