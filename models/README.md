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

Due to GitHub's file size limitations (100 MB), model weights are hosted externally.

### Option 1: Direct Download
*[Download link coming soon]*

### Option 2: Using gdown (Google Drive)
```bash
pip install gdown
gdown "YOUR_GOOGLE_DRIVE_LINK" -O models/densenet201_best.h5
```

### Option 3: Hugging Face Hub
```python
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="YOUR_USERNAME/building-classification",
    filename="densenet201_best.h5"
)
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
