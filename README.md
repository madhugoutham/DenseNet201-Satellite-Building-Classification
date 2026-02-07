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
- 📊 **K-Fold Cross Validation**: Robust model evaluation with 5-fold cross-validation
- 🗺️ **Geographic Coverage**: Data from DeKalb County and Cook County, Illinois
- 📐 **512×512 Image Patches**: High-resolution aerial imagery from NAIP

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
│   ├── 01_data_collection/      # Data acquisition scripts
│   │   ├── 01_patch_csv_generator.ipynb
│   │   ├── 02_image_patch_downloading.ipynb
│   │   ├── 03_download_buildings_by_zipcode.ipynb
│   │   ├── 04_data_collector.ipynb
│   │   └── 05_image_collection.ipynb
│   │
│   ├── 02_model_training/       # Model training notebooks
│   │   ├── 01_densenet201_training.ipynb
│   │   ├── 02_cross_validation_training.ipynb
│   │   ├── 03_model_classification.ipynb
│   │   └── 04_building_classification_model.ipynb
│   │
│   └── 03_inference/            # Prediction & evaluation
│       ├── 01_building_prediction.ipynb
│       ├── 02_model_evaluation.ipynb
│       └── 03_new_prediction.ipynb
│
├── data/
│   ├── processed/               # Organized image dataset
│   │   ├── train/               # Training images (~70%)
│   │   ├── val/                 # Validation images (~15%)
│   │   └── test/                # Test images (~15%)
│   │
│   └── metadata/                # CSV metadata files
│       ├── buildings_metadata.csv
│       └── output_predictions.csv
│
├── models/                      # Trained model weights
│   └── README.md                # Model download instructions
│
└── results/                     # Evaluation results
    ├── figures/
    └── confusion_matrices/
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for training)

### Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/building-classification.git
cd building-classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running Inference

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
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0

# Predict
predictions = model.predict(img_array)
predicted_class = classes[np.argmax(predictions)]
print(f"Predicted class: {predicted_class}")
```

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **Overall Accuracy** | TBD |
| **Weighted F1-Score** | TBD |
| **Macro F1-Score** | TBD |

*Detailed confusion matrices and per-class metrics available in `results/`*

---

## 📥 Trained Models

Due to file size limitations, trained model weights are hosted externally:

| Model | Description | Download |
|-------|-------------|----------|
| `densenet201_best.h5` | Best performing model | [Coming Soon] |
| `densenet201_fold_*.h5` | K-fold cross-validation models | [Coming Soon] |

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

- NAIP (National Agriculture Imagery Program) for aerial imagery
- Microsoft Building Footprints for building polygons
- TensorFlow/Keras team for DenseNet201 implementation

---

## 📧 Contact

For questions or collaboration inquiries, please open an issue or contact the authors.
