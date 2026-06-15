# Advanced Text Mining Frameworks for Automated Feedback Analysis

### An Empirical Study on Amazon Fine Food Reviews
**University of Milano-Bicocca** **Academic Year:** 2025/2026  
**Authors:** Junaid Ahmed (923714), Summan Gul (925663)  
**Course:** Text Mining and Search  

---

## Project Overview
This project addresses the challenges of extracting operational insights from large-scale, imbalanced customer feedback datasets. Using a subset of 20,000 product reviews from the Amazon Fine Food repository, the framework pairs supervised sentiment classification with unsupervised anomaly detection. 

The core architecture evaluates statistical frequency encodings against dense neural vectors to map sentiment polarity while leveraging geometric clustering to uncover underlying supply chain and logistics failures automatically.

---

## ⚙️ System Architecture & Methodology

The pipeline processes unstructured text through two major analytical components:

### 1. Preprocessing & Bifurcated Pipeline
* **Pipeline A (Statistical/TF-IDF Support):** Fully processes text using case standardization, regex noise cleansing, NLTK stopword elimination, and WordNet morphological lemmatization to maximize token density.
* **Pipeline B (Neural/Word2Vec Support):** Maintains word order and grammatical context by bypassing stopword pruning and lemmatization, preserving critical sequence anchors required for neural embeddings.

### 2. Feature Extraction & Modeling
* **Supervised Layer:** Evaluates a high-dimensional sparse **TF-IDF Vectorizer** (with unigrams/bigrams) against a dense, 100-dimensional **Word2Vec (CBOW)** mean embedding model. Sentiment classification is executed via **Logistic Regression** optimized with an inverse class-weighting parameter to address class imbalance.
* **Unsupervised Layer:** Uses **K-Means Clustering** ($k=4$) to partition reviews into operational themes, mapped visually using **t-SNE Spatial Projections** to differentiate between product traits (e.g., flavor profiles) and logistics defects (e.g., packaging damage).

---

## Empirical Results

* **Peak Classification Accuracy:** **89.62%** achieved by the TF-IDF framework.
* **Performance Premium:** TF-IDF outperformed Word2Vec by **3.13%**, exposing a *Sentiment Dilution Effect* where geometric mean pooling in Word2Vec smooths out high-impact keywords (e.g., *terrible*, *disappointed*).
* **Unsupervised Resolution:** K-Means clustering successfully isolated supply chain metadata clusters (keywords: *box*, *package*, *received*) from baseline product evaluations without requiring manual labeling loops.

---

## Directory Structure

```text
├── data/
│   └── raw/                          # Amazon Fine Food Reviews raw data reference
├── notebooks/
│   ├── 1_data_cleaning_integrity.ipynb # Data validation and duplicate pruning
│   ├── 2_bifurcated_preprocessing.ipynb # NLP pipelines (Pipeline A & Pipeline B)
│   ├── 3_supervised_sentiment.ipynb    # TF-IDF vs Word2Vec + Logistic Regression
│   └── 4_unsupervised_clustering.ipynb # K-Means partitioning & t-SNE projections
├── src/
│   ├── preprocessing.py               # Custom text transformation functions
│   └── modeling.py                    # Training and vector evaluation utilities
├── README.md                          # Repository documentation
└── requirements.txt                   # Dependency configurations
