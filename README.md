# Foundations of Deep Learning Project: Multi-Class Image Classification and Capacity Regularization on CIFAR-10

This project applies advanced Deep Learning principles and Convolutional Neural Network (CNN) architectures to execute robust computer vision tasks. By utilizing the standard CIFAR-10 benchmark dataset, the project designs, optimizes, and evaluates multi-class classification models from scratch and cross-compares their performance against a state-of-the-art pre-trained transfer learning baseline (ResNet50) under strict computational constraints.

---

## Overview

The project focuses on the implementation and structural analysis of deep image recognition networks, executing two core computational tasks:

1. **Supervised Image Classification:** Building an end-to-end predictive pipeline to automatically categorize low-resolution target images into 10 distinct object classes, showcasing scalable solutions for visual pattern recognition.
2. **Capacity Regularization Study:** Analyzing how layer scaling, network depth, and structural regularizers (such as Dropout) mitigate model overfitting on small input matrices.

---

## Datasets

### CIFAR-10 Dataset

- **Source:** [Keras Datasets API / Canadian Institute for Advanced Research](https://www.cs.toronto.edu/~kriz/cifar.html)
- **Description:** A curated, balanced collection of tiny RGB color images representing standard natural and structural objects.
- **Variables Include:**
  - **Instances:** 60,000 total images (50,000 training, 10,000 testing).
  - **Input Dimensions:** $32 \times 32$ pixels with 3 color channels (RGB).
  - **Target Classes:** 10 categories (`airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`).

---

## Methodology

### 1. Data Pipeline and Stratification
- **Reproducibility Configuration:** Locked global execution environment seeds across Python, NumPy, and TensorFlow to ensure perfectly consistent, deterministic metrics.
- **Dynamic Slicing Setup:** Replicated production folder parsing behavior by concatenating raw arrays and cutting an 80/20 training split. The remaining balance was systematically partitioned into distinct **Validation** and **Testing** datasets using a strict mathematical batch slicing layout (`take` and `skip`).
- **Label Formatting:** Converted target integers into categorical tracking matrices via **One-Hot Encoding** to align with multi-class loss dimensions.

### 2. Model Development and Evolution
Developed three distinct custom convolutional models from scratch to investigate how capacity levels impact validation stability:
- **Model 1 (Light Baseline):** Deployed a minimal baseline containing two consecutive convolutional blocks with standard MaxPooling to establish raw convergence rates.
- **Model 2 (Intermediate CNN):** Introduced a third convolutional layer with broader feature channels and integrated a spatial `Dropout(0.2)` layer to limit initial over-fitting trends.
- **Model 3 (Deep Regularized):** Scaled layer depth up to 128 channels combined with progressive Dropout filters (`0.2`, `0.3`, and `0.5`) across dense layers to penalize semantic co-adaptation on image noise.

### 3. Optimized Transfer Learning Baseline
- **The Overhead Challenge:** Traditional transfer learning upsamples $32 \times 32$ pixels to the native ImageNet input dimensions ($224 \times 224$), creating heavy multi-dimensional matrices that result in training times exceeding 12 hours on standard Colab hardware.
- **The Native Shape Solution:** Bypassed the spatial upsampling block and initialized a pre-trained **ResNet50** backbone configured to accept raw $32 \times 32 \times 3$ tensors directly. The core feature extraction base was completely frozen (`trainable = False`), and a custom dense classification head was attached on top to achieve rapid metric convergence in just minutes.

### 4. Evaluation and Performance Analysis
- **Learning Curve Visualization:** Tracked and plotted loss and accuracy metrics over training intervals to analyze convergence trends.
- **Classification Performance:** Generated full multi-class classification reports detailing Precision, Recall, and F1-Scores across all 10 evaluation classes on unseen data.
- **Error Analysis Matrix:** Plotted absolute heatmaps using Confusion Matrices to locate specific class boundary confusions.

---

## Usage

1. **Install Dependencies:** Ensure your Python environment has the necessary deep learning modules configured:
   ```bash
   pip install tensorflow scikit-learn matplotlib seaborn numpy pandas
