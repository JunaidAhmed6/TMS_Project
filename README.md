# Advanced Text Mining Frameworks for Automated Feedback Analysis

### Empirical Study on Amazon Fine Food Reviews
**University of Milano-Bicocca** | **Academic Year:** 2025/2026  
**Authors:** Junaid Ahmed (923714), Summan Gul (925663)  
**Course:** Text Mining and Search  

---

## Project Overview
This project extracts operational, supply chain, and sentiment insights from large-scale, imbalanced customer feedback datasets. Using a curated subset of 20,000 product reviews from the Amazon Fine Food repository, the framework pairs supervised sentiment classification with unsupervised anomaly detection. 

The core architecture benchmarks statistical frequency encodings against dense neural vectors to map sentiment polarity, while leveraging geometric clustering to isolate underlying supply chain and logistical failures automatically without human-annotated labeling loops.

---

## Datasets

### Amazon Fine Food Reviews

- **Source:** [Amazon Fine Food Reviews Dataset](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
- **Description:** A collection of customer reviews for food products sold on Amazon.
- **Variables Include:**
  - **Id:** Row index identifier.
  - **ProductId:** Unique identifier for the product (ASIN).
  - **UserId:** Unique identifier for the customer.
  - **ProfileName:** Display name of the reviewer.
  - **HelpfulnessNumerator / HelpfulnessDenominator:** Metrics indicating how many users found the review helpful.
  - **Score:** Original numeric rating assigned to the product (1–5 stars).
  - **Time:** Timestamp of the review submission (Unix time).
  - **Summary:** A brief headline summarizing the user's feedback.
  - **Text:** The full plaintext content of the customer review.

---

## Core Architecture & Methodology

The pipeline processes unstructured text through two major analytical paths:

### 1. Preprocessing & Bifurcated Pipeline
* **Pipeline A (Statistical / TF-IDF Support):** Optimizes token density for the sparse framework using case standardization, regex noise cleansing (stripping HTML tags like `<br/>`), standard NLTK English stopword elimination, and WordNet morphological lemmatization.
* **Pipeline B (Neural / Word2Vec Support):** Protects local semantic context and grammatical layout. It maintains word sequences and includes high-frequency stopwords as contextual anchors, which are necessary for the sliding window neural embedding model.

### 2. Feature Extraction & Modeling
* **Supervised Layer:** Evaluates a high-dimensional sparse **TF-IDF Vectorizer** (max 2,500 columns, unigram/bigram range) against a dense, 100-dimensional **Word2Vec (CBOW)** mean embedding model. Sentiment classification is executed via **Logistic Regression** optimized with an inverse class-weighting parameter to handle the 83.60% positive class imbalance.
* **Unsupervised Layer:** Uses **K-Means Clustering** ($k=4$) to partition reviews into operational themes, mapped visually using **t-SNE Spatial Projections** to differentiate between product traits (e.g., flavor profiles) and logistics defects (e.g., packaging damage).

---

## Empirical Results
* **Peak Classification Accuracy:** **89.62%** achieved by the TF-IDF framework.
* **Performance Premium:** TF-IDF outperformed Word2Vec by **3.13%**, exposing a *Sentiment Dilution Effect* where geometric mean pooling across variable document lengths smooths out high-impact keywords (e.g., *terrible*, *disappointed*, *return*).
* **Unsupervised Resolution:** K-Means clustering successfully isolated supply chain metadata clusters (keywords: *box*, *package*, *order*, *received*) from baseline product quality evaluations.

---

## Requirements & Dependencies

Ensure you have Python 3.10+ installed. The project relies on the following external libraries:

* **pandas** (>= 2.0.0) For data manipulation and tabular alignment.
* **numpy** (>= 1.24.0)  For vectorized array calculations and matrix operations.
* **nltk** (>= 3.8.1) For tokenization, stopword filtering, and lemmatization.
* **gensim** (>= 4.3.0)  For training and parsing the Word2Vec CBOW embedding space.
* **scikit-learn** (>= 1.2.0)  For TF-IDF vectorization, Logistic Regression, K-Means clustering, and evaluation metrics.
* **scipy** (>= 1.10.0)  For scientific calculations supporting downstream model optimization.
* **matplotlib** (>= 3.7.0)  For baseline plot generations.
* **seaborn** (>= 0.12.0)  For visualizing high-dimensional t-SNE spatial manifolds.
