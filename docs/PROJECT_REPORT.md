# College Assignment Project Report: Mini Project 9

# NewsLens AI — News Article Classification Using Machine Learning

**Course:** B.Tech / Academic Project Submission  
**Subject:** Natural Language Processing & Machine Learning Laboratory  
**Developer / Student Name:** Daniel Richard Xavier  
**Date of Completion:** October 28, 2026  
**Repository:** [https://github.com/drx2k06/miniproject9.git](https://github.com/drx2k06/miniproject9.git)  

---

## Abstract
With the exponential growth of online digital journalism, news portals and aggregation engines process hundreds of thousands of news dispatches daily. Manual categorization into topical sections is unscalable, costly, and subject to human fatigue. This project presents **NewsLens AI**, a complete supervised text classification system that automatically analyzes news headlines and body text to predict their thematic category across four standard domains: **World**, **Sports**, **Business**, and **Sci/Tech**.

The system utilizes the established academic **AG News** benchmark dataset, evaluating both **Multinomial Naive Bayes** and **Logistic Regression** classifiers operating over sublinear **TF-IDF** (Term Frequency - Inverse Document Frequency) feature representations with unigrams and bigrams. Model selection is performed strictly using 5-fold Stratified Cross-Validation on the training partition. On the unseen official held-out test set of 7,600 articles, the champion Logistic Regression model achieves **89.97% Accuracy**, a **Macro F1-Score of 0.8993**, and **0.8992 Macro Precision**. The project incorporates a production-grade Streamlit web interface for interactive inference, comprehensive automated test suites (36/36 tests passing), error analysis, and educational explainability cards designed for viva examinations.

---

## 1. Introduction
Text classification is one of the foundational challenges in Natural Language Processing (NLP). It involves assigning predefined categorical labels to unstructured natural language text based on its lexical, semantic, and syntactic patterns.

In modern journalistic workflows, automated topic tagging powers:
- Real-time content feeds on news aggregator platforms (Google News, Apple News).
- Sectional routing in digital newsrooms (routing wire reports to sports, business, or foreign bureaus).
- Reader personalization, semantic search, and content recommendation systems.
- Media monitoring, public policy tracking, and business intelligence.

---

## 2. Problem Definition
To design, implement, evaluate, and deploy a machine learning text-classification application that accepts an English news headline, summary, or article body and accurately classifies it into one of four verified categories:
1. **World** (International politics, diplomacy, treaties, conflicts)
2. **Sports** (Matches, championships, athletic scores, leagues)
3. **Business** (Finance, stock markets, macroeconomics, corporate news)
4. **Sci/Tech** (Science, computing, software, cybersecurity, space)

### Key Constraints & Challenges:
- **No Data Leakage:** Feature extraction parameters (vocabulary, inverse document frequencies) must be learned exclusively from the training split.
- **CPU Friendliness:** The pipeline must execute in under 10 seconds without requiring GPUs, cloud infrastructure, or paid inference APIs.
- **Handling Ambiguity:** Articles spanning overlapping domains (e.g., Apple quarterly earnings or Olympic security treaties) must produce meaningful probability distributions.
- **Clarification of Model Scope:** The system classifies topic domain; it is explicitly not a fact-checker or fake-news detector.

---

## 3. Project Objectives
1. **Data Ingestion & Validation:** Acquire the verified AG News benchmark dataset and perform an automated quality audit (missing values, class balance, text length distributions).
2. **Robust Text Preprocessing:** Implement an NLP cleaner that unescapes HTML entities, normalizes whitespace, and preserves critical domain acronyms and technical terms.
3. **Model Training & Comparison:** Implement and compare a generative probabilistic model (Multinomial Naive Bayes) against a discriminative linear model (Logistic Regression).
4. **Empirical Model Selection:** Perform 5-fold Stratified Cross-Validation on the training set to choose the champion model without contaminating the test set.
5. **Rigorous Evaluation:** Measure Accuracy, Macro Precision, Macro Recall, Macro F1, Weighted F1, per-class metrics, and confusion matrix on 7,600 unseen test samples.
6. **Interactive Deployment:** Build a modern, multi-page Streamlit web application providing single-article classification, probability distribution bar charts, and a dataset explorer.
7. **Quality Assurance:** Validate the implementation with a Pytest automated test suite covering unit, integration, and smoke tests.

---

## 4. Dataset Description: AG News Benchmark
The system utilizes the **AG News** classification corpus compiled by Xiang Zhang, Junbo Zhao, and Yann LeCun (NeurIPS 2015).

- **Total Corpus Size:** 127,600 news dispatches across 4 balanced categories.
- **Experimental Training Partition:** 24,000 stratified samples (6,000 per class, random seed 42) ensuring fast, reproducible local training while preserving statistical fidelity.
- **Held-Out Test Partition:** 7,600 full benchmark articles (1,900 per class).
- **Class Labels:**
  - `0`: **World**
  - `1`: **Sports**
  - `2`: **Business**
  - `3`: **Sci/Tech**

---

## 5. System Methodology & Algorithms

### 5.1 Text Cleaning & Preprocessing
Raw news text often contains web formatting artifacts. The preprocessor applies:
- HTML entity translation (`&amp;` $\to$ `&`, `&quot;` $\to$ `"`).
- HTML markup excision (`<p>`, `<div>`, `<b>`).
- URL removal (`https://...`).
- Control character and carriage escape cleansing (`\b`, `\n`, `\t`).
- Whitespace collapsing and bounding trimming.

### 5.2 TF-IDF Vectorization
Text documents are vectorized into a 25,000-dimensional numerical space using sublinear TF-IDF:
$$\text{TF-IDF}(t, d, D) = (1 + \log(\text{TF}(t, d))) \times \left(\log\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|} + 1\right)$$
- **N-Grams:** Unigrams and bigrams $(1, 2)$ capture phrase context (e.g., "Wall Street", "prime minister", "artificial intelligence").
- **Sublinear TF:** Prevents bursty word repetitions from artificially inflating feature weights.
- **L2 Normalization:** Normalizes vector magnitudes to 1, removing bias from document length variations.

### 5.3 Candidate Classification Algorithms

#### Multinomial Naive Bayes (Baseline)
Naive Bayes applies Bayes' Theorem with the assumption that all features are conditionally independent given the class:
$$P(c \mid \mathbf{x}) \propto P(c) \prod_{j=1}^{V} P(x_j \mid c)^{x_j}$$
Laplace smoothing ($\alpha = 0.1$) handles unseen terms in the vocabulary.

#### Logistic Regression (Selected Champion)
Logistic Regression models the posterior probability of each class using the softmax activation function over linear hyperplanes:
$$P(y = c \mid \mathbf{x}) = \frac{\exp(\mathbf{w}_c^T \mathbf{x} + b_c)}{\sum_{k=0}^{K-1} \exp(\mathbf{w}_k^T \mathbf{x} + b_k)}$$
The model minimizes cross-entropy loss penalized by L2 regularization ($C = 1.0$) via the L-BFGS numerical optimizer.

---

## 6. Experimental Results & Discussion

### 6.1 Test Set Performance Comparison

| Metric | Multinomial Naive Bayes | Logistic Regression (Champion) |
| :--- | :---: | :---: |
| **Accuracy** | 89.86% | **89.97%** |
| **Macro Precision** | 0.8983 | **0.8992** |
| **Macro Recall** | 0.8986 | **0.8997** |
| **Macro F1-Score** | 0.8982 | **0.8993** |
| **Weighted F1-Score** | 0.8982 | **0.8993** |
| **5-Fold CV Macro F1** | 0.9009 (±0.0028) | **0.9028 (±0.0023)** |
| **Training Time (24k items)** | **4.23 seconds** | 5.62 seconds |

### 6.2 Per-Class Breakdown (Champion Model)

| Category | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **World** | 0.9109 | 0.9037 | 0.9073 | 1,900 |
| **Sports** | **0.9406** | **0.9747** | **0.9574** | 1,900 |
| **Business** | 0.8753 | 0.8500 | 0.8625 | 1,900 |
| **Sci/Tech** | 0.8701 | 0.8705 | 0.8703 | 1,900 |
| **Macro Average** | **0.8992** | **0.8997** | **0.8993** | **7,600** |

### 6.3 Confusion Matrix
Out of 7,600 unseen test articles:
- **Sports** achieved the highest accuracy, with 1,852 out of 1,900 articles classified correctly (97.47% recall).
- **World** achieved 1,717 correct classifications (90.37% recall).
- The primary source of misclassification occurred along the **Business $\leftrightarrow$ Sci/Tech** boundary:
  - 190 Business articles were predicted as Sci/Tech.
  - 141 Sci/Tech articles were predicted as Business.
  *Root Cause:* Technology companies (Apple, Microsoft, Google, Intel) frequently appear in financial news covering quarterly revenue, share repurchases, and antitrust litigation, confounding lexical cues.

---

## 7. Automated Testing & Verification
The project incorporates a test suite written with `pytest`:
- **Test Modules:** `test_preprocessing.py`, `test_model.py`, `test_prediction.py`, `test_smoke.py`.
- **Total Tests:** 36 automated test cases.
- **Passing Rate:** **100% (36 passed in 6.22 seconds)**.
- **Coverage Areas:** Text cleaning, whitespace normalization, HTML entity unescaping, length truncation, model fitting, probability calibration, pipeline serialization, inference accuracy, schema validation, and artifact persistence.

---

## 8. Viva Voce Frequently Asked Questions (Exam Guide)

### Q1: Why use TF-IDF instead of simple word counts (Bag of Words)?
**Answer:** Bag of Words weights terms purely by raw frequency, meaning generic words (e.g., "said", "today", "year") dominate the representation. TF-IDF penalizes terms that appear frequently across the entire corpus while rewarding terms that are specific to a small subset of documents, giving much higher weights to domain-discriminative vocabulary like "quarterback" or "semiconductor".

### Q2: Why use Sublinear TF scaling (`sublinear_tf=True`)?
**Answer:** Without sublinear scaling, an article mentioning the word "software" 20 times would receive a term-frequency weight 20 times higher than an article mentioning it once. Using $1 + \log(\text{tf})$ dampens extreme frequencies, reflecting diminishing marginal information return.

### Q3: Why is Macro F1 preferred over simple Accuracy?
**Answer:** While our test set is balanced (1,900 samples per class), in real-world scenarios class distributions are often highly skewed (e.g., far more World news than Sci/Tech). Simple accuracy can be artificially inflated by predicting the majority class. Macro F1 calculates the unweighted arithmetic mean of F1 scores across all classes:
$$\text{Macro F1} = \frac{1}{K} \sum_{k=1}^{K} \text{F1}_k$$
This guarantees that each class contributes equally to the final metric.

### Q4: What is the difference between Generative and Discriminative models in this project?
**Answer:**
- **Multinomial Naive Bayes is Generative:** It models the joint probability distribution $P(\mathbf{x}, y) = P(y) P(\mathbf{x} \mid y)$ and uses Bayes' Theorem to invert it into a posterior probability.
- **Logistic Regression is Discriminative:** It directly models the conditional probability distribution $P(y \mid \mathbf{x})$ by fitting linear hyperplanes that separate classes in feature space.

### Q5: How was data leakage prevented?
**Answer:** Data leakage was prevented by encapsulating the preprocessor, the `TfidfVectorizer`, and the classifier inside a single `sklearn.pipeline.Pipeline`. The vectorizer was fitted exclusively on training data. The 7,600 test samples were strictly held out until the final evaluation step.

---

## 9. Limitations
1. **Bag-of-Words Limitation:** TF-IDF captures lexical frequencies and local n-gram pairs, but does not capture long-range contextual relationships or polysemy (words with multiple meanings in different contexts).
2. **Fixed Category Taxonomy:** The system supports the 4 categories present in AG News. News articles falling outside these categories (e.g., Entertainment, Health, Lifestyle) will be mapped to the closest of the 4 available categories.
3. **Domain Classification vs. Fact-Checking:** The system classifies topics based on lexical patterns; it cannot determine whether an article is factually true, misleading, or satirical.

---

## 10. Conclusion & Future Scope
NewsLens AI successfully demonstrates that lightweight, CPU-friendly supervised machine learning pipelines can achieve high classification accuracy (>89.9%) on standardized text benchmarks. The system combines software engineering best practices, data integrity, comprehensive automated testing, and an interactive Streamlit user interface.

### Future Scope:
- Incorporating fine-tuned Transformer models (e.g., DistilBERT or RoBERTa) for deep semantic representation.
- Expanding the category hierarchy to include Entertainment, Health, and Politics sub-sections.
- Supporting multilingual news classification across Spanish, French, and Hindi feeds.

---

## 11. References
1. Zhang, X., Zhao, J., & LeCun, Y. (2015). Character-level Convolutional Networks for Text Classification. *NeurIPS 2015*.
2. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University Press.
3. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
4. Gulli, A. (2005). AG's Corpus of News Articles. *Academic Repository*.
