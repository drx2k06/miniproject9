# NewsLens AI — News Article Classification Using Machine Learning

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![Tests Passing](https://img.shields.io/badge/tests-36%2F36%20passing-brightgreen.svg)]()

**College Assignment Mini Project 9**  
**Developer:** Daniel Richard Xavier  
**Destination Repository:** [https://github.com/drx2k06/miniproject9.git](https://github.com/drx2k06/miniproject9.git)  
**Local Project Directory:** `C:\Users\drx20\Projects\miniproject9`  

---

## Table of Contents
1. [Abstract](#1-abstract)
2. [Introduction](#2-introduction)
3. [Problem Statement](#3-problem-statement)
4. [Objectives](#4-objectives)
5. [Existing Approach and Proposed Solution](#5-existing-approach-and-proposed-solution)
6. [Dataset and Citation](#6-dataset-and-citation)
7. [Technology Stack](#7-technology-stack)
8. [Text Preprocessing](#8-text-preprocessing)
9. [TF-IDF Feature Extraction](#9-tf-idf-feature-extraction)
10. [Naive Bayes Algorithm](#10-naive-bayes-algorithm)
11. [Logistic Regression Algorithm](#11-logistic-regression-algorithm)
12. [System Architecture](#12-system-architecture)
13. [Installation Instructions](#13-installation-instructions)
14. [Training and Evaluation Commands](#14-training-and-evaluation-commands)
15. [Streamlit Execution](#15-streamlit-execution)
16. [Actual Results and Visualizations](#16-actual-results-and-visualizations)
17. [Limitations](#17-limitations)
18. [Future Enhancements](#18-future-enhancements)
19. [References](#19-references)
20. [Developer Contributions](#20-developer-contributions)

---

## 1. Abstract
In the modern information ecosystem, thousands of news dispatches are published every hour across global news wires. Manually sorting these unstructured texts into sectional categories is slow, inconsistent, and unscalable. **NewsLens AI** is a complete, production-grade text classification system built with Natural Language Processing (NLP) and supervised Machine Learning. Operating on the official **AG News** benchmark corpus, the system evaluates and compares **Multinomial Naive Bayes** and **Logistic Regression** over sublinear **TF-IDF** feature vectors with unigrams and bigrams. Model selection is performed strictly on the training partition via 5-fold Stratified Cross-Validation. On the untouched held-out benchmark test set of 7,600 news articles, the champion Logistic Regression pipeline achieves **89.97% Accuracy**, a **Macro F1-Score of 0.8993**, and **0.8992 Macro Precision**. The project includes a multi-page interactive Streamlit web dashboard, a 36-test automated validation suite, and assignment reports designed for college viva examinations.

---

## 2. Introduction
Digital news portals, search aggregators, and media monitoring services require automated classification to route articles to specialized news desks, power topical search filters, and serve personalized reader feeds. Text classification maps unstructured language strings into discrete categorical buckets based on semantic and lexical features.

NewsLens AI addresses the categorization of news into four principal journalistic domains:
- 🌍 **World:** Bilateral summits, diplomacy, United Nations resolutions, peace treaties, foreign affairs.
- ⚽ **Sports:** Match fixtures, athletic championships, tournament results, league transfers, Olympic events.
- 💼 **Business:** Stock market fluctuations, macroeconomic indices, corporate mergers, central banking, quarterly earnings.
- 🔬 **Sci/Tech:** Space exploration, computing advancements, artificial intelligence, software developments, cybersecurity vulnerabilities.

---

## 3. Problem Statement
Given an arbitrary news headline $h$, an article summary/body $b$, or their combination $x = (h, b)$, develop a machine learning classifier $f(x)$ that accurately predicts the most probable category label $y \in \{\text{World}, \text{Sports}, \text{Business}, \text{Sci/Tech}\}$ and produces calibrated class probabilities.

### Critical Engineering Requirements:
1. **Zero Data Leakage:** Vectorizer parameters and vocabulary must be fitted solely on training data using an atomic `scikit-learn Pipeline`.
2. **CPU-Friendly Execution:** Training, inference, and testing must execute efficiently on standard student laptops without GPUs or paid APIs.
3. **No Synthetic Data Substitution:** Models must be trained and verified on verified public academic benchmark data.
4. **Transparent Scope:** The system must clarify that it categorizes topic domains based on lexical patterns; it does not evaluate factual accuracy or detect misinformation.

---

## 4. Objectives
- Ingest and validate the AG News benchmark dataset (schema, missing values, duplicates, class distributions).
- Design and implement a reusable text preprocessing pipeline that cleans HTML entities and whitespace while preserving informative domain entities and numbers.
- Build scikit-learn Pipelines combining TF-IDF feature extraction with Multinomial Naive Bayes and Logistic Regression.
- Perform 5-fold Stratified Cross-Validation for model selection without contaminating the held-out test split.
- Measure comprehensive test metrics: Accuracy, Macro Precision, Macro Recall, Macro F1, Weighted F1, per-class metrics, and confusion matrix.
- Build an interactive Streamlit web application featuring article classification, confidence estimates, probability bar charts, model evaluation metrics, and an exploratory data viewer.
- Validate the system with an automated test suite achieving 100% passing tests.

---

## 5. Existing Approach and Proposed Solution

| Dimension | Traditional Approach | NewsLens AI Solution |
| :--- | :--- | :--- |
| **Categorization** | Manual editorial tagging or keyword rule lists. | Supervised Machine Learning with statistical generalization. |
| **Feature Extraction** | Unweighted Bag-of-Words (raw counts). | Sublinear TF-IDF with $(1, 2)$ word n-grams and vocabulary limits. |
| **Evaluation Strategy** | Ad-hoc single train/test split. | 5-Fold Stratified Cross-Validation + pristine held-out test benchmark. |
| **Model Selection** | Pre-selected arbitrary choice. | Empirical selection based on measured Macro F1 and probability calibration. |
| **Data Leakage** | Vectorizer fit on combined dataset. | Strict `sklearn.pipeline.Pipeline` fitting only on training data. |
| **User Interface** | Terminal scripts only. | Interactive multi-page Streamlit dashboard with preset test articles. |

---

## 6. Dataset and Citation
NewsLens AI is trained and evaluated on the official **AG News** classification benchmark.

- **Citation:**
  > Zhang, X., Zhao, J., & LeCun, Y. (2015). Character-level Convolutional Networks for Text Classification. *Advances in Neural Information Processing Systems (NeurIPS 2015)*, 28, 649–657.
- **Source:** Open benchmark derived from Antonio Gulli's AG Corpus of news articles indexed across 2,000+ sources.
- **Dataset Dimensions:**
  - Full Benchmark: 120,000 training articles + 7,600 held-out test articles.
  - Reproducible Project Train Sample: 24,000 articles (stratified, 6,000 per class, random seed 42).
  - Unseen Test Split: 7,600 articles (1,900 per class).
- **Categories:**
  - `0`: World
  - `1`: Sports
  - `2`: Business
  - `3`: Sci/Tech

---

## 7. Technology Stack
- **Programming Language:** Python 3.11 / 3.12 (CPU-optimized)
- **Data Manipulation:** Pandas 3.0, NumPy 2.2, PyArrow 26.0
- **Machine Learning & NLP:** Scikit-Learn 1.9, Joblib 1.6
- **Web Application:** Streamlit 1.66
- **Visualization:** Matplotlib 3.11, Seaborn 0.13
- **Automated Testing:** Pytest 8.3
- **Version Control:** Git & GitHub

---

## 8. Text Preprocessing
Implemented in `src/preprocessing.py`, the preprocessing pipeline systematically removes syndication artifacts while preserving informative terms:
1. **HTML Entity Decoding:** Translates escaped codes (e.g., `&amp;` $\to$ `&`, `&quot;` $\to$ `"`, `&lt;` $\to$ `<`).
2. **Markup & URL Cleansing:** Removes HTML tags (`<p>`, `<b>`) and web addresses (`http://...`).
3. **Control Character & Glitch Removal:** Strips control bytes and feed escape artifacts (`\b`, `\n`, `\t`).
4. **Whitespace Normalization:** Collapses consecutive whitespace characters into a single space and trims bounding whitespace.
5. **Entity & Acronym Preservation:** Does not apply destructive stemming, allowing acronyms (`NASA`, `NASDAQ`, `UEFA`) and technical terms (`.NET`, `iPhone-15`, `3.5%`) to retain distinct token forms.

---

## 9. TF-IDF Feature Extraction
The text vectorizer transforms unstructured tokens into normalized vectors in $\mathbb{R}^{25000}$:

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

1. **Sublinear Term Frequency (`sublinear_tf=True`):**
   $$\text{TF}(t, d) = 1 + \log(f_{t, d}) \quad \text{for } f_{t, d} > 0$$
   Mitigates the distortion caused by frequent repeated words within an article.
2. **Inverse Document Frequency with Smoothing:**
   $$\text{IDF}(t, D) = \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$
   Penalizes words appearing uniformly across all documents and rewards category-specific terms.
3. **N-Grams:** Captures unigrams and bigrams $(1, 2)$ to preserve phrase meaning ("Wall Street", "prime minister").
4. **L2 Normalization:** Normalizes all document vectors to unit length $\|\mathbf{v}\|_2 = 1$.

---

## 10. Naive Bayes Algorithm
**Multinomial Naive Bayes** serves as the probabilistic baseline model. It applies Bayes' rule under the conditional independence assumption:

$$P(y = c \mid \mathbf{x}) \propto P(y = c) \prod_{j=1}^{V} P(x_j \mid y = c)^{x_j}$$

- **Laplace Smoothing ($\alpha = 0.1$):**
  $$\hat{\theta}_{c, j} = \frac{N_{c, j} + \alpha}{N_c + \alpha V}$$
- **Characteristics:** Fast training (4.23s on 24k samples), highly competitive baseline (89.86% test accuracy).

---

## 11. Logistic Regression Algorithm
**Logistic Regression** serves as the linear discriminative model and was selected as the **Champion Model**.

For multi-class classification with $K = 4$ classes, it computes linear decision boundaries and converts them into posterior probabilities via the softmax function:

$$P(y = c \mid \mathbf{x}) = \frac{\exp(\mathbf{w}_c^T \mathbf{x} + b_c)}{\sum_{k=0}^{K-1} \exp(\mathbf{w}_k^T \mathbf{x} + b_k)}$$

- **Loss Function:** Minimizes L2-regularized multinomial cross-entropy loss:
  $$\mathcal{L}(\mathbf{w}) = - \sum_{i=1}^{N} \sum_{k=0}^{K-1} \mathbf{1}\{y_i = k\} \log P(y = k \mid \mathbf{x}_i) + \frac{1}{2C} \sum_{k=0}^{K-1} \|\mathbf{w}_k\|_2^2$$
- **Hyperparameters:** $C = 1.0$, solver = `lbfgs`, `max_iter = 500`.
- **Advantages:** Produces smooth, calibrated probabilities and handles correlated n-gram features effectively.

---

## 12. System Architecture

```
User Input (Headline & Description)
               │
               ▼
┌──────────────────────────────────────────────┐
│       src/preprocessing.py                   │
│  - HTML Entity Unescaping (&amp; -> &)       │
│  - Tag & URL Stripping                       │
│  - Whitespace Normalization                  │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│       scikit-learn Pipeline                  │
│  ┌────────────────────────────────────────┐  │
│  │ TfidfVectorizer (ngram=(1, 2), 25k)   │  │
│  └───────────────────┬────────────────────┘  │
│                      │                       │
│                      ▼                       │
│  ┌────────────────────────────────────────┐  │
│  │ LogisticRegression (C=1.0, Softmax)    │  │
│  └────────────────────────────────────────┘  │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│  Structured Output:                          │
│  - Predicted Category (World/Sports/etc.)    │
│  - Confidence Estimate (%)                   │
│  - 4-Class Probability Distribution Chart     │
│  - Text Statistics & Explainability Notes    │
└──────────────────────────────────────────────┘
```

---

## 13. Installation Instructions

### Prerequisites
- Python 3.11 or Python 3.12 installed.
- Git installed.

### Clone the Repository
```bash
git clone https://github.com/drx2k06/miniproject9.git
cd miniproject9
```

### Create and Activate Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 14. Training and Evaluation Commands

### Train Models and Select Champion
```bash
python train.py
```
*Executes data loading, 5-fold cross-validation, candidate model fitting, test evaluation, plot generation, and saves `models/newslens_pipeline.joblib`.*

### Standalone Evaluation
```bash
python evaluate.py
```
*Loads the saved fitted pipeline and computes full classification report, confusion matrix, and exports `outputs/evaluation_summary.json`.*

### Run Automated Tests
```bash
python -m pytest tests/ -v
```
*Executes all 36 unit, integration, and smoke tests.*

---

## 15. Streamlit Execution

To launch the web dashboard:
```bash
streamlit run app.py
```
The browser will automatically open at `http://localhost:8501`.

### Streamlit Features:
- **Page A (Overview):** Project motivation, pipeline architecture, viva quick reference guide, and developer credentials.
- **Page B (Classify an Article):** Interactive headline and body input, 4 one-click category presets, real-time prediction badge, confidence score, horizontal probability bar chart, and disclaimers.
- **Page C (Model Evaluation):** Live metric cards, side-by-side model comparison table, confusion matrix plot, per-class breakdown, and error analysis.
- **Page D (Dataset Explorer):** Dataset dimensions, category distributions, character/word length histograms, interactive article search, and data quality metrics.

---

## 16. Actual Results and Visualizations

### Benchmark Comparison on Unseen Held-Out Test Set (7,600 Articles)

| Metric | Multinomial Naive Bayes | Logistic Regression (Champion) |
| :--- | :---: | :---: |
| **Accuracy** | 89.86% | **89.97%** |
| **Macro Precision** | 0.8983 | **0.8992** |
| **Macro Recall** | 0.8986 | **0.8997** |
| **Macro F1-Score** | 0.8982 | **0.8993** |
| **Weighted F1-Score** | 0.8982 | **0.8993** |
| **5-Fold CV Macro F1** | 0.9009 (±0.0028) | **0.9028 (±0.0023)** |
| **Training Time** | **4.23 s** | 5.62 s |

### Per-Class Performance Breakdown (Champion Model)

| Category | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **World** | 0.9109 | 0.9037 | 0.9073 | 1,900 |
| **Sports** | **0.9406** | **0.9747** | **0.9574** | 1,900 |
| **Business** | 0.8753 | 0.8500 | 0.8625 | 1,900 |
| **Sci/Tech** | 0.8701 | 0.8705 | 0.8703 | 1,900 |
| **Macro Average** | **0.8992** | **0.8997** | **0.8993** | **7,600** |

### Confusion Matrix Heatmap
The generated confusion matrix is saved in `outputs/confusion_matrix.png`:
- **Sports** achieved the highest accuracy (1,852 / 1,900 correct, 97.47% recall).
- **World** achieved 1,717 / 1,900 correct (90.37% recall).
- Cross-category overlap was concentrated between **Business** and **Sci/Tech** (190 Business articles predicted as Sci/Tech, 141 Sci/Tech articles predicted as Business).

---

## 17. Limitations
1. **Lexical Representation:** TF-IDF relies on n-gram co-occurrence and does not model contextual syntax or word polysemy.
2. **Fixed Category Set:** The model maps inputs exclusively to one of the 4 AG News categories. Out-of-domain articles (e.g., Entertainment or Health) will be assigned to the closest existing category.
3. **No Factual Verification:** NewsLens AI predicts thematic domain; it does not verify article veracity or detect fake news.

---

## 18. Future Enhancements
- Integration of fine-tuned Transformer encoders (DistilBERT or RoBERTa) for deep contextual embeddings.
- Expanding category coverage to include Entertainment, Health, and Environment desks.
- Multilingual classification support across European and Asian news feeds.
- Integration of an automated API endpoint via FastAPI alongside the Streamlit UI.

---

## 19. References
1. Zhang, X., Zhao, J., & LeCun, Y. (2015). Character-level Convolutional Networks for Text Classification. *NeurIPS 2015*.
2. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825-2830.
3. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.
4. Gulli, A. (2005). AG's Corpus of News Articles. *Academic Repository*.

---

## 20. Developer Contributions
- **Sole Developer & Author:** **Daniel Richard Xavier**
- **Responsibilities:**
  - Designed and implemented end-to-end NLP and ML architecture.
  - Implemented data loading, schema validation, and text preprocessing modules.
  - Built scikit-learn pipelines for Multinomial Naive Bayes and Logistic Regression.
  - Evaluated models using 5-fold Stratified Cross-Validation and held-out test sets.
  - Designed and developed the 4-page Streamlit web dashboard.
  - Wrote 36 automated unit and smoke tests using Pytest.
  - Authored comprehensive assignment reports, viva guide, and project documentation.
