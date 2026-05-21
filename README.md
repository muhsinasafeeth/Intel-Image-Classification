# 🖼️ Intel Image Classification — CNN from Scratch

[![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://tensorflow.org)
[![Kaggle Dataset](https://img.shields.io/badge/Dataset-Intel%20Image%20Classification-20BEFF?logo=kaggle)](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A complete end-to-end deep learning project that designs, trains, and evaluates a **Convolutional Neural Network (CNN) built entirely from scratch** using TensorFlow/Keras. The model classifies natural scene images into 6 categories from the Intel Image Classification dataset. This project covers every stage of a real deep learning pipeline — from raw data download to visual prediction analysis.

---

## 📌 Problem Statement

Given a natural scene image of size 150×150 pixels, the goal is to correctly identify which of the following **6 scene categories** it belongs to:

| Class | Description |
|-------|-------------|
| 🏙️ `buildings` | Urban architecture, structures, cityscapes |
| 🌲 `forest` | Dense tree canopies and woodland scenes |
| 🧊 `glacier` | Ice sheets, snow fields, frozen landscapes |
| ⛰️ `mountain` | Rocky terrain, hillsides, elevated landforms |
| 🌊 `sea` | Ocean, sea, and open water scenes |
| 🛣️ `street` | Road-level views, streets, sidewalks |

This is a **multi-class image classification** problem — each image belongs to exactly one class.

---

## 🗂️ Repository Structure

```
intel-image-classification/
│
├── Intel_Classification_Assignment.ipynb   # Main notebook — full pipeline
├── README.md                               # Project documentation (this file)
├── pipeline_overview.html                  # Visual one-page pipeline poster
├── pipeline_overview.pdf                   # PDF version of the pipeline poster
│
└── assets/
    └── pipeline_overview.png               # Pipeline diagram screenshot
```

---

## 🔄 End-to-End Pipeline

```
Kaggle Dataset Download
        │
        ▼
 Data Exploration ───────► Class names, sample images per class
        │
        ▼
 Preprocessing & Augmentation
 (Rescale + Rotation + Zoom + Flip + Shift)
        │
        ▼
 CNN Architecture
 Conv(32) → Conv(64) → Conv(128) → Dense(256) → Softmax(6)
        │
        ▼
 Model Compilation
 Adam + Categorical Crossentropy + Accuracy
        │
        ▼
 Training  ──► EarlyStopping + ModelCheckpoint
        │
        ▼
 Evaluation
 ├── Accuracy / Loss Curves
 ├── Test Set Accuracy & Loss
 ├── Classification Report (Precision, Recall, F1)
 ├── Confusion Matrix Heatmap
 └── Visual Prediction Samples (10 images)
```

---

## 📦 Part A — Data Understanding & Preprocessing

### Dataset

The **Intel Image Classification** dataset is downloaded directly from Kaggle using the Kaggle API. It contains approximately **25,000 natural scene images** across 6 classes, each image being an RGB photograph.

- **Source**: [`puneet6060/intel-image-classification`](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
- **Training folder**: `seg_train/seg_train/` — used for training and validation
- **Test folder**: `seg_test/seg_test/` — used only for final evaluation

### Exploration

After loading, the class names are extracted by listing subdirectories of the training folder. A 2×3 grid of sample images is displayed — one randomly selected image per class — to visually confirm the dataset structure and appreciate the variation within each scene type.

### Preprocessing

All images are resized to a fixed **150×150 pixels** using Keras `ImageDataGenerator`. Pixel values are rescaled from the range [0, 255] to [0, 1] by dividing by 255 (`rescale=1./255`). This normalization step ensures stable and faster gradient updates during training.

### Data Augmentation (Training Set Only)

To improve the model's ability to generalize to unseen images and reduce overfitting, the following augmentations are applied **randomly during training** — they are never applied to the test set:

| Augmentation | Value | Purpose |
|---|---|---|
| `rotation_range` | 20° | Handles tilted/rotated scene captures |
| `zoom_range` | 0.2 | Handles varying distances from subject |
| `width_shift_range` | 0.2 | Handles horizontal framing differences |
| `height_shift_range` | 0.2 | Handles vertical framing differences |
| `horizontal_flip` | True | Mirror flips for symmetric scenes (e.g. sea, mountain) |

### Dataset Split

The dataset is split into three subsets:

- **Training set** — 80% of `seg_train`, used to update model weights
- **Validation set** — 20% of `seg_train` (via `validation_split=0.2`), used to monitor training and trigger early stopping
- **Test set** — the entire `seg_test` folder, used only once at the end for final unbiased evaluation

---

## 🧠 Part B — CNN Model Development

### Architecture Design

The CNN is built using Keras `Sequential` API with **3 convolutional blocks** followed by a fully connected classifier head. The filter count doubles with each block (32 → 64 → 128) to progressively detect more complex and abstract visual features — edges and textures in early layers, shapes and patterns in deeper ones.

```
Input Image: (150, 150, 3)
│
├── [Block 1]
│     Conv2D(32 filters, 3×3, ReLU)   → Feature maps: 148×148×32
│     MaxPooling2D(2×2)               → Feature maps:  74×74×32
│
├── [Block 2]
│     Conv2D(64 filters, 3×3, ReLU)   → Feature maps:  72×72×64
│     MaxPooling2D(2×2)               → Feature maps:  36×36×64
│
├── [Block 3]
│     Conv2D(128 filters, 3×3, ReLU)  → Feature maps:  34×34×128
│     MaxPooling2D(2×2)               → Feature maps:  17×17×128
│
├── Flatten()                         → 36,992 values
│
├── Dense(256, ReLU)                  → 256 neurons
│
├── Dropout(0.5)                      → Randomly drops 50% of neurons during training
│
└── Dense(6, Softmax)                 → 6 probability scores (one per class)
```

**Why this design?**
- **ReLU** activation introduces non-linearity without suffering from vanishing gradients
- **MaxPooling** reduces spatial dimensions, lowering computation and adding translation invariance
- **Dropout(0.5)** is a powerful regularization technique — during training it randomly deactivates half the neurons in the Dense layer, forcing the network to learn redundant representations and preventing over-reliance on specific neurons
- **Softmax** output converts raw scores into probabilities that sum to 1, perfect for multi-class classification

### Compilation

```python
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

- **Adam optimizer** — combines the benefits of momentum and adaptive learning rates, making it robust and fast-converging for image classification tasks
- **Categorical Crossentropy** — the standard loss function for multi-class problems with one-hot encoded labels; it penalizes confident wrong predictions heavily
- **Accuracy** — the primary metric tracked during training and reported at evaluation

### Training

The model is trained for up to **20 epochs** with two callbacks that work together to ensure the best possible model is saved:

**EarlyStopping**
```python
EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
```
Monitors the validation loss at the end of every epoch. If it does not improve for 5 consecutive epochs, training is stopped early and the model weights are restored to the epoch where validation loss was lowest. This prevents the model from overfitting on the training data.

**ModelCheckpoint**
```python
ModelCheckpoint('best_cnn_model.h5', monitor='val_accuracy', save_best_only=True)
```
Saves the model weights to disk only when the validation accuracy improves. This guarantees that the best-performing version of the model across all epochs is always preserved, even if training continues past the peak.

---

## 📊 Part C — Model Evaluation & Interpretation

### Training Curves

After training, accuracy and loss curves are plotted side by side to visually diagnose model behaviour:

- **Accuracy curve** — tracks `train_accuracy` vs `val_accuracy` epoch by epoch. Ideally both rise together and converge. A large gap where training accuracy is much higher than validation accuracy indicates overfitting.
- **Loss curve** — tracks `train_loss` vs `val_loss`. Both should decrease and stabilize near similar values. A rising validation loss while training loss keeps falling is a classic sign of overfitting.

### Test Set Evaluation

The best saved model (`best_cnn_model.h5`) is evaluated on the held-out test set — images that were never seen during training or validation. This gives the most honest, unbiased measure of real-world performance.

```python
test_loss, test_accuracy = model.evaluate(test_generator)
```

### Classification Report

Using `sklearn.metrics.classification_report`, per-class metrics are printed for all 6 categories:

| Metric | What it measures |
|---|---|
| **Precision** | Of all images predicted as class X, what fraction actually were class X? |
| **Recall** | Of all actual class X images, what fraction did the model correctly identify? |
| **F1-Score** | Harmonic mean of Precision and Recall — a balanced single metric per class |
| **Support** | Total number of true instances of that class in the test set |

### Confusion Matrix

A heatmap confusion matrix is plotted using Seaborn. Rows represent **actual classes**, columns represent **predicted classes**.

- Diagonal cells (top-left to bottom-right) show **correct predictions** — higher is better
- Off-diagonal cells reveal **misclassifications** — for example, `glacier` being confused with `mountain` is a common and interpretable error since both feature cold, pale, rocky imagery

### Visual Prediction Samples

10 random images are sampled from the test set. For each image, both the **true label** and the **model's predicted label** are displayed. This qualitative step complements the quantitative metrics by showing exactly what kinds of images the model handles well and where it struggles.

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

Get your API token from [kaggle.com → Account → Create New Token](https://www.kaggle.com/settings/account). This downloads a `kaggle.json` file. Then run:

```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 4. Run the notebook

Open `Intel_Classification_Assignment.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab and run all cells top to bottom.

> **Tip:** If running on Google Colab, upload your `kaggle.json` directly — the notebook handles credential setup programmatically.

---

## 🔑 Key Techniques Summary

| Technique | Why it was used |
|---|---|
| **Data Augmentation** | Artificially increases training diversity to reduce overfitting |
| **Batch Normalization (optional)** | Stabilizes training and allows higher learning rates |
| **Dropout (0.5)** | Regularization — prevents neurons from co-adapting |
| **EarlyStopping** | Stops training before the model overfits |
| **ModelCheckpoint** | Ensures the best model across all epochs is saved |
| **Classification Report** | Gives per-class granular performance beyond just accuracy |
| **Confusion Matrix** | Reveals which class pairs are systematically confused |

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Test Accuracy | *(run notebook to see)* |
| Test Loss | *(run notebook to see)* |

> Full per-class precision, recall, and F1-scores are printed in the notebook output. Results will vary slightly due to random augmentation and weight initialization.

---

## 📚 References

- [Intel Image Classification Dataset — Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)
- [TensorFlow/Keras Sequential API](https://www.tensorflow.org/api_docs/python/tf/keras/Sequential)
- [Keras ImageDataGenerator](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator)
- [Scikit-learn Classification Report](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)
- [Understanding Dropout — Srivastava et al. 2014](https://jmlr.org/papers/v15/srivastava14a.html)

---

## 👤 Author

**Muhsina** — Deep Learning Assignment   


---

*Built with TensorFlow 2.x · Intel Image Classification Dataset · Kaggle*
## 👤 Author

**Muhsina** — Deep Learning Assignment  


---

*Built with TensorFlow 2.x and ❤️*
