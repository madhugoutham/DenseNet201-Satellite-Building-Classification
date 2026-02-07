# Building Classification using DenseNet201

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Deep learning-based building classification from aerial/satellite imagery using DenseNet201 transfer learning. This repository contains the code, sample dataset, and trained models accompanying our research paper.

---

## 🏗️ Overview

This project presents a **DenseNet201-based convolutional neural network** for classifying buildings from aerial imagery into **7 distinct categories**:

| Category | Description |
|----------|-------------|
| **Commercial** | Retail stores, shopping centers, office buildings |
| **High-rise** | Multi-story residential/commercial towers |
| **Hospital** | Healthcare facilities |
| **Industrial** | Factories, warehouses, manufacturing plants |
| **Multi-family** | Apartments, condominiums, townhouses |
| **Schools** | Educational institutions |
| **Single-family** | Detached residential homes |

### Key Features

- 🔬 **Transfer Learning**: Pre-trained DenseNet201 backbone fine-tuned for building classification
- 🌐 **Google Earth Data**: 512×512 pixel images at ~0.15 m/pixel resolution via samgeo
- 🏛️ **7 Building Classes**: Comprehensive taxonomy covering major urban building types
- 🔧 **Segmentation Pipeline**: ReFineNet + watershed algorithm for building extraction

---

## 📂 Repository Structure

```
building-classification/
├── README.md                    # This file
├── LICENSE                      # MIT License
├── CITATION.cff                 # Citation metadata
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
│
├── paper/                       # Research paper
│   └── Building_Classification_Research_Paper.docx
│
├── notebooks/
│   ├── 01_data_collection.ipynb        # Satellite image acquisition via samgeo
│   ├── 02_preprocessing_segmentation.ipynb  # ReFineNet + morphological ops
│   ├── 03_model_training.ipynb         # DenseNet201 training with paper hyperparams
│   └── 04_evaluation_inference.ipynb   # Metrics, confusion matrix, predictions
│
├── data/
│   ├── processed/               # Organized image dataset
│   │   ├── train/               # Training images (80%)
│   │   ├── val/                 # Validation images (10%)
│   │   └── test/                # Test images (10%)
│   │
│   └── metadata/                # CSV metadata files
│
├── models/                      # Trained model weights
│   └── README.md                # Model download instructions
│
└── results/                     # Evaluation results & figures
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for training)

### Installation

```bash
# Clone the repository
git clone https://github.com/madhugoutham/building-classification.git
cd building-classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running Notebooks

1. **Data Collection**: `notebooks/01_data_collection.ipynb`
2. **Preprocessing**: `notebooks/02_preprocessing_segmentation.ipynb`
3. **Training**: `notebooks/03_model_training.ipynb`
4. **Evaluation**: `notebooks/04_evaluation_inference.ipynb`

### Quick Inference

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load trained model
model = load_model('models/densenet201_best.h5')

# Class labels
classes = ['Commercial', 'High', 'Hospital', 'Industrial', 'Multi', 'Schools', 'Single']

# Load and preprocess image
img = image.load_img('path/to/building.tif', target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
predictions = model.predict(img_array)
predicted_class = classes[np.argmax(predictions)]
confidence = np.max(predictions) * 100
print(f"Predicted: {predicted_class} ({confidence:.1f}%)")
```

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Overall Test Accuracy** | 84.40% |
| **Validation Accuracy** | 84.39% |
| **Macro F1-Score** | 0.84 |
| **Weighted F1-Score** | 0.84 |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Commercial | 0.80 | 0.60 | 0.69 | 20 |
| High-rise | 0.95 | 0.90 | 0.92 | 20 |
| Hospital | 0.84 | 0.80 | 0.82 | 20 |
| Industrial | 0.83 | 0.95 | 0.89 | 21 |
| Multi-family | 0.77 | 0.85 | 0.81 | 20 |
| Schools | 0.77 | 0.85 | 0.81 | 20 |
| Single-family | 0.95 | 0.95 | 0.95 | 20 |

---

## 🧠 Model Architecture

**Hyperparameters (Table 4 in paper):**

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam (β1=0.9, β2=0.999) |
| Learning Rate | 1e-4 (reduced on plateau) |
| Batch Size | 32 |
| Epochs | Up to 20 (early stopping) |
| Dropout Rate | 0.5 |
| L2 Regularization | 0.001 |

---

## 📥 Trained Models

Model weights are hosted externally due to file size:

| Model | Description | Download |
|-------|-------------|----------|
| `densenet201_best.h5` | Best performing model | [Coming Soon] |

See `models/README.md` for download instructions.

---

## 📚 Citation

If you use this code or dataset in your research, please cite:

```bibtex
@article{author2025building,
  title={Building Classification from Aerial Imagery using DenseNet201},
  author={Author Name},
  journal={Journal Name},
  year={2025}
}
```

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Google Earth for satellite imagery
- [segment-geospatial (samgeo)](https://github.com/opengeos/segment-geospatial) for image acquisition
- TensorFlow/Keras team for DenseNet201 implementation

---

## 📧 Contact

For questions or collaboration inquiries, please open an issue or contact the authors.
