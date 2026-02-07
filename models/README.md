# Trained Models

This directory contains trained DenseNet201 model weights for building classification.

## Model Information

| Model | Description | Input Size | Classes |
|-------|-------------|------------|---------|
| `densenet201_best.h5` | Best performing model | 224×224×3 | 7 |

## Model Architecture

- **Base Model**: DenseNet201 (pretrained on ImageNet)
- **Top Layers**: Global Average Pooling → Dense(256, ReLU) → Dropout(0.5) → Dense(7, Softmax)
- **Fine-tuning**: Last 100 layers of DenseNet201 are trainable

## Download Instructions

Due to GitHub's file size limitations (100 MB), model weights are hosted on Zenodo.

### 📥 Download via Zenodo (Recommended)
You can download the best performing model directly from Zenodo:

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18513710.svg)](https://doi.org/10.5281/zenodo.18513710)

1. Go to the [Zenodo Record](https://doi.org/10.5281/zenodo.18513710).
2. Download `DenseNet201_Best_Model.h5` (138 MB).
3. Place the file in this `models/` directory.

### using Python script
Run the provided script to download automatically:
```bash
python download_model.py
```

## Usage

```python
from tensorflow.keras.models import load_model

# Load model
model = load_model('models/densenet201_best.h5')

# Print model summary
model.summary()
```

## Class Labels

| Index | Class |
|-------|-------|
| 0 | Commercial |
| 1 | High (High-rise) |
| 2 | Hospital |
| 3 | Industrial |
| 4 | Multi (Multi-family) |
| 5 | Schools |
| 6 | Single (Single-family) |
