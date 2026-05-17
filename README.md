# Text Mining Project: Sentiment Analysis and Feature Importance on Amazon Fine Food Reviews

This project applies advanced Natural Language Processing (NLP) techniques and Machine Learning models to extract valuable insights from consumer-generated e-commerce data. By analyzing customer feedback from Amazon, the project aims to classify review sentiments and isolate key linguistic patterns using both statistical and semantic text representations.

---

## Overview

The project focuses on the comprehensive text mining of food product reviews, executing two core computational tasks:

1. **Supervised Sentiment Classification:** Building a predictive pipeline to automatically categorize product reviews into positive and negative sentiments, providing a scalable solution for monitoring customer satisfaction.
2. **Feature Importance Identification:** Analyzing the structural parameters of the trained models to isolate the specific terms that most heavily influence classification decisions.

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

## Project Aims

- **Extract Actionable Insights:** Identify primary customer preferences and structural patterns in consumer feedback within the food and beverage retail domain.
- **Comparative Representation Framework:** Evaluate and contrast traditional frequency-based statistical representations (**TF-IDF**) against modern semantic word vector embeddings (**Word2Vec**).
- **Sentiment Classification:** Convert raw star ratings into clear binary sentiments (Negative: 1–2 stars; Positive: 4–5 stars) and train classification models to predict user sentiment.
- **Feature Importance Analysis:** Isolate and analyze the top informative linguistic markers (coefficients) that explicitly drive positive or negative classification choices.

---

## Methodology

### 1. Data Loading and Preprocessing
- **Data Sampling:** Subsampled a computationally efficient partition of 20,000 instances from the raw source dataset.
- **Label Engineering:** Extracted explicit positive and negative classes by filtering out neutral 3-star reviews to ensure clean classification boundaries.
- **Text Cleaning Pipeline:** - Standardized all strings to lowercase.
  - Parsed out formatting noise and raw structural elements like HTML tags via regular expressions.
  - Stripped non-alphabetic symbols, numbers, and punctuation signs.
- **Linguistic Normalization:** Tokenized textual blocks into word units, eliminated non-informative words using the standard NLTK stopword lexicon, and applied **Lemmatization** to collapse words to their valid baseline dictionary roots.

### 2. Feature Extraction and Representation
- **TF-IDF Vectorization:** Built a sparse frequency matrix restricted to the top 2,500 maximum features, weighting words based on their specific importance across documents.
- **Word Embeddings (Word2Vec):** Trained a dense 100-dimensional continuous vector space model over review tokens using a local context window of 5 words. Complete document vectors were compiled by averaging constituent word vectors.

### 3. Classification and Evaluation
- **Supervised Learning Model:** Deployed a **Logistic Regression** classifier over an 80/20 train-test validation split across both feature sets.
- **Performance Evaluation:** Quantified and cross-compared model predictive effectiveness utilizing exact accuracy metrics.

---

## Results and Insights

- **Classification Metrics:** - **TF-IDF Representation:** Achieved a high classification accuracy of **89.62%** ($0.8962$).
  - **Word2Vec Representation:** Achieved a classification accuracy of **86.49%** ($0.8649$).
  - **Key Comparative Finding:** The frequency-weighted TF-IDF matrix yielded superior classification precision over the dense average Word2Vec embedding vectors for this text corpus.
- **Top Informative Features:** - Strongest negative textual indicators identified include words like *disappointed*, *terrible*, *horrible*, *return*, and *weak*.
  - Strongest positive textual indicators identified include words like *great*, *love*, *best*, *delicious*, and *good*.

---

## Usage

1. **Install Dependencies:** Ensure your Python environment has the necessary modules configured:
   ```bash
   pip install gensim nltk scikit-learn pandas numpy
