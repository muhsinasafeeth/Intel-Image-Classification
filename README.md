# 🖼️ Intel Image Classification — CNN from Scratch

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://tensorflow.org)
[![Kaggle Dataset](https://img.shields.io/badge/Dataset-Intel%20Image%20Classification-20BEFF?logo=kaggle)](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A deep learning assignment that builds a **Convolutional Neural Network (CNN) from scratch** to classify natural scene images into 6 categories using the Intel Image Classification dataset.

---

## 📌 Problem Statement

Classify 150×150 pixel natural scene images into one of **6 categories**:

| Class | Description |
|-------|-------------|
| 🏙️ `buildings` | Urban architecture |
| 🌲 `forest` | Dense tree cover |
| 🧊 `glacier` | Ice and snow landscapes |
| ⛰️ `mountain` | Mountain terrain |
| 🌊 `sea` | Ocean and water bodies |
| 🛣️ `street` | Roads and street-level scenes |

---

## 🗂️ Repository Structure

```
intel-image-classification/
│
├── Intel_Classification_Assignment.ipynb   # Main notebook (end-to-end pipeline)
├── README.md                               # Project overview (this file)
├── pipeline_overview.html                  # Visual walkthrough of the pipeline
│
└── assets/
    └── pipeline_overview.png               # Pipeline diagram for README
```

---

## ⚙️ Pipeline Overview

```
Raw Dataset
    │
    ▼
Data Exploration ──────────► Visualize 6 Classes
    │
    ▼
Preprocessing & Augmentation
(Rescale + Rotation + Zoom + Flip)
    │
    ▼
CNN Architecture
Conv2D(32) → Conv2D(64) → Conv2D(128) → Dense(256) → Softmax(6)
    │
    ▼
Training (EarlyStopping + ModelCheckpoint)
    │
    ▼
Evaluation ──► Accuracy/Loss Curves
            ├── Classification Report
            ├── Confusion Matrix
            └── Visual Prediction Samples
```

---

## 🏗️ Model Architecture

```
Input: (150, 150, 3)
│
├── Conv2D(32, 3×3, ReLU) → MaxPool(2×2)
├── Conv2D(64, 3×3, ReLU) → MaxPool(2×2)
├── Conv2D(128, 3×3, ReLU) → MaxPool(2×2)
│
├── Flatten
├── Dense(256, ReLU)
├── Dropout(0.5)
│
└── Dense(6, Softmax)  ← Output
```

| Hyperparameter | Value |
|----------------|-------|
| Image Size | 150 × 150 |
| Batch Size | 32 |
| Optimizer | Adam |
| Loss | Categorical Crossentropy |
| Max Epochs | 20 |
| Early Stopping Patience | 5 |

---

## 📦 Dataset

- **Source**: [Kaggle — Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
- **Total Images**: ~25,000
- **Split**: Training / Validation (80/20 from `seg_train`) / Test (`seg_test`)
- **Image Size**: 150 × 150 × 3 (RGB)

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/intel-image-classification.git
cd intel-image-classification
```

### 2. Install dependencies

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn kaggle
```

### 3. Set up Kaggle credentials

Place your `kaggle.json` API token in `~/.kaggle/`:

```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 4. Run the notebook

Open and run `Intel_Classification_Assignment.ipynb` top to bottom in Jupyter or Google Colab.

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Test Accuracy | *(run notebook to see)* |
| Test Loss | *(run notebook to see)* |

> Per-class precision, recall, and F1-scores are printed via `sklearn`'s `classification_report`. A heatmap confusion matrix is also generated.

---

## 🔑 Key Techniques Used

- **Data Augmentation** — rotation, zoom, flips to improve generalization
- **Early Stopping** — stops training when validation loss stagnates
- **Model Checkpointing** — saves the best weights during training
- **Dropout (0.5)** — regularization to reduce overfitting
- **Confusion Matrix + Classification Report** — thorough multi-class evaluation

---

## 📚 References

- [Intel Image Classification Dataset — Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
- [TensorFlow/Keras Documentation](https://www.tensorflow.org/api_docs)
- [Scikit-learn Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)

---

## 👤 Author

**Muhsina** — Deep Learning Assignment  


---

*Built with TensorFlow 2.x and ❤️*
