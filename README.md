# Bearing Fault Detection

A machine learning and deep learning project for **bearing fault
detection and diagnosis** using vibration time-series data.

The project explores both traditional machine learning approaches on
engineered signal features and deep learning approaches that learn
patterns directly from vibration signals and their image
representations.

## Overview

Bearings are critical components in rotating machinery, and faults such
as inner-race, outer-race, and ball defects can lead to machine failure
if they are not detected early.

This project investigates multiple approaches for identifying bearing
conditions from vibration signals:

-   Statistical and signal-based feature extraction
-   Traditional machine learning classifiers
-   Gramian Angular Field (GAF/GADF) image representations
-   2D Convolutional Neural Networks (CNNs)
-   Temporal Convolutional Networks (TCNs)

The work is organized into separate stages so that different
representations and models can be compared.

## Dataset

The experiments use vibration data from the **Case Western Reserve
University (CWRU) Bearing Data Center** dataset.

The vibration signals are divided into fixed-length windows before
feature extraction or deep-learning processing.

Typical classes used in the project include:

-   Normal
-   Ball fault
-   Inner-race fault
-   Outer-race fault

Some experiments further divide the data into multiple
operating-condition/fault categories.

## Project Structure

``` text
Bearing-Fault-Detection/
│
├── 2D_CNN/
│   └── 2D CNN experiments and trained model files
│
├── Data/
│   └── Dataset files and processed data
│
├── Models/
│   └── Machine learning models and model-analysis files
│
├── Output/
│   └── Generated outputs, processed datasets, and results
│
├── TCN/
│   └── Temporal Convolutional Network experiments
│
├── src/
│   └── Source code and supporting implementation
│
└── LICENSE
```

## Methodology

### 1. Signal Preprocessing

The raw vibration signal is divided into fixed-size windows.

Depending on the experiment, preprocessing can include filtering and
signal transformation before the data is passed to a model.

A typical processing pipeline is:

``` text
Raw Vibration Signal
        │
        ▼
Signal Preprocessing
        │
        ▼
Windowing
        │
        ├──────────────► Statistical Features ──► ML Models
        │
        ▼
Signal-to-Image Transformation
        │
        ▼
GAF / GADF Images
        │
        ▼
2D CNN
```

### 2. Feature-Based Machine Learning

For the traditional machine learning experiments, statistical and
time-domain features are extracted from each signal window.

Features explored include:

-   Mean
-   Standard deviation
-   Maximum absolute value
-   RMS
-   Kurtosis
-   Skewness
-   Peak-to-peak value
-   Crest factor
-   Shape factor
-   Impulse factor
-   Clearance factor

These features are used with different machine learning algorithms for
fault classification.

Models explored include:

-   Random Forest
-   K-Nearest Neighbors (KNN)
-   XGBoost
-   Gradient Boosting
-   Isolation Forest

Feature analysis, including **SHAP analysis**, is also included for
selected models to understand feature contributions.

### 3. GAF / GADF Representation

To apply image-based deep learning to vibration signals, one-dimensional
time-series windows are transformed into two-dimensional images using
**Gramian Angular Fields**.

The project explores:

-   GAF
-   GADF

The resulting images allow a 2D CNN to learn spatial patterns generated
from the original temporal signal.

### 4. 2D CNN

A custom **2D Convolutional Neural Network** is trained on the generated
GAF/GADF images.

The general pipeline is:

``` text
Vibration Signal
      │
      ▼
Preprocessing
      │
      ▼
Fixed-Length Window
      │
      ▼
GAF / GADF
      │
      ▼
2D Image
      │
      ▼
CNN
      │
      ▼
Fault Class
```

The `2D_CNN` directory contains the experiments, training outputs, and
saved model weights associated with this approach.

### 5. Temporal Convolutional Network (TCN)

The project also investigates **Temporal Convolutional Networks (TCNs)**
as an alternative to recurrent architectures for learning temporal
dependencies directly from vibration signals.

TCNs use:

-   Causal convolutions
-   Dilated convolutions
-   Residual connections

This allows the network to capture patterns across different temporal
scales while processing the sequence efficiently.

The `TCN` directory contains the related experiments and processed
files.

## Models Compared

The repository therefore contains experiments across multiple types of
approaches:

  Approach            Input                  Purpose
  ------------------- ---------------------- ----------------------------------
  Random Forest       Engineered features    Fault classification
  KNN                 Engineered features    Fault classification
  XGBoost             Engineered features    Fault classification
  Gradient Boosting   Engineered features    Fault classification
  Isolation Forest    Signal features        Anomaly detection
  2D CNN              GAF/GADF images        Image-based fault classification
  TCN                 1D vibration signals   Temporal fault classification

## Goals

The main objectives of the project are:

1.  Detect bearing faults from vibration signals.
2.  Compare traditional machine learning with deep learning approaches.
3.  Investigate different representations of time-series data.
4.  Study whether converting 1D vibration signals into 2D images can
    provide useful features for CNN-based classification.
5.  Explore temporal modelling using TCNs.
6.  Analyze model predictions and important features where applicable.

## Results

The repository contains the datasets, processed outputs, trained models,
and experiments used to evaluate the different approaches.

Results may vary depending on:

-   Dataset split
-   Window size
-   Preprocessing method
-   Fault categories
-   Model architecture
-   Training configuration

For reproducible comparisons, the corresponding experiment code and
saved outputs should be used together.

## Technologies

-   **Python**
-   **PyTorch**
-   **scikit-learn**
-   **XGBoost**
-   **NumPy**
-   **Pandas**
-   **SciPy**
-   **Matplotlib**
-   **SHAP**
-   Gramian Angular Field representations
-   Temporal Convolutional Networks
-   Convolutional Neural Networks

## Future Work

Possible extensions include:

-   Multi-scale TCN architectures
-   Attention mechanisms
-   CNN + TCN hybrid architectures
-   Improved signal denoising and preprocessing
-   More operating-condition variations
-   Cross-condition and cross-load evaluation
-   Explainability for deep learning models
-   Early/pre-fault detection rather than only fault classification

## Project Status

The repository is an ongoing research/learning project. Different
approaches are being implemented and evaluated progressively, with the
current work focusing on comparing **2D CNN-based signal representations
with temporal models such as TCNs**.

## License

This project is distributed under the license included in the
repository.
