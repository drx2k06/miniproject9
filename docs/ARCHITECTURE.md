# System Architecture: NewsLens AI

**Developer:** Daniel Richard Xavier  
**Application:** News Article Topic Classification Engine  

---

## 1. High-Level Component Architecture

```
+-------------------------------------------------------------------------------+
|                               Streamlit UI (app.py)                           |
|  [Page A: Overview] [Page B: Classify Article] [Page C: Evaluation] [Page D: Explorer]
+-------------------------------------------------------------------------------+
                                      |
                                      v
+-------------------------------------------------------------------------------+
|                       Prediction Interface (src/prediction.py)               |
|  - Input Validation (Length checks, Empty strings, Char limits)               |
|  - Headline + Body Concatenation                                              |
|  - Category Confidence and Multi-Class Probability Calculations               |
+-------------------------------------------------------------------------------+
                                      |
                                      v
+-------------------------------------------------------------------------------+
|                  Serialized Model Pipeline (models/newslens_pipeline.joblib)  |
|                                                                               |
|   +-----------------------------------------------------------------------+   |
|   | Step 1: Text Preprocessor (src/preprocessing.py)                     |   |
|   |         - HTML Entity Unescaping (&amp; -> &)                         |   |
|   |         - Tag & URL Stripping (<p>, https://)                         |   |
|   |         - Control Character & Glitch Removal (\b, \t, \n)             |   |
|   |         - Whitespace Normalization & Trimming                         |   |
|   +-----------------------------------------------------------------------+   |
|                                     |                                         |
|                                     v                                         |
|   +-----------------------------------------------------------------------+   |
|   | Step 2: TF-IDF Vectorizer (scikit-learn TfidfVectorizer)             |   |
|   |         - N-Gram Range: (1, 2) unigrams and bigrams                   |   |
|   |         - Max Features: 25,000 vocabulary                             |   |
|   |         - Sublinear TF scaling (1 + log(tf))                          |   |
|   |         - English stop word elimination                               |   |
|   |         - L2 Unit Normalization                                       |   |
|   +-----------------------------------------------------------------------+   |
|                                     |                                         |
|                                     v                                         |
|   +-----------------------------------------------------------------------+   |
|   | Step 3: Champion Classifier (scikit-learn LogisticRegression)         |   |
|   |         - Multi-Class Softmax Decision Layer                          |   |
|   |         - L2 Weight Penalty (C=1.0)                                   |   |
|   |         - L-BFGS Solver Optimization                                  |   |
|   +-----------------------------------------------------------------------+   |
+-------------------------------------------------------------------------------+
                                      |
                     +----------------+----------------+
                     |                                 |
                     v                                 v
+------------------------------------+  +---------------------------------------+
|    Structured Prediction Result     |  |          Evaluation & Audit           |
|  - Predicted Category              |  |  - outputs/metrics.json               |
|  - Confidence Score (%)            |  |  - outputs/confusion_matrix.png       |
|  - 4-Class Probabilities Array     |  |  - outputs/model_comparison.png       |
|  - Article Statistics (chars/words)|  |  - outputs/error_analysis.json        |
+------------------------------------+  +---------------------------------------+
```

---

## 2. Directory Layout and Module Responsibilities

```
miniproject9/
├── app.py                     # Streamlit multi-page web application
├── train.py                   # Automated training, CV, model selection, and artifact generator
├── evaluate.py                # Standalone test evaluation script
├── requirements.txt           # Project Python dependencies
├── .gitignore                 # Version control filtering rules
├── README.md                  # Comprehensive GitHub project guide
│
├── src/                       # Core modular source library
│   ├── __init__.py            # Package root & developer metadata
│   ├── data_loader.py         # AG News benchmark download, loading, schema validation
│   ├── preprocessing.py       # Deterministic text cleaning, HTML unescaping, validation
│   ├── model.py               # Pipeline construction, training, CV, joblib serialization
│   ├── prediction.py          # Real-time inference engine and confidence estimator
│   └── evaluation.py          # Metric calculations, confusion matrix, plot generation
│
├── models/
│   └── newslens_pipeline.joblib # Compressed, portable fitted pipeline artifact (1.01 MB)
│
├── outputs/
│   ├── metrics.json           # Comprehensive JSON record of CV and test metrics
│   ├── confusion_matrix.png   # 300 DPI publication heatmap of test classifications
│   ├── model_comparison.png   # Visual comparison bar chart (NB vs LR)
│   ├── error_analysis.json    # Detailed misclassified test items for inspection
│   └── evaluation_summary.json# Formatted summary from evaluate.py
│
├── data/
│   ├── ag_news_train_sample.parquet # Stratified 24,000-article training benchmark
│   └── ag_news_test.parquet         # Official 7,600-article held-out test split
│
├── tests/
│   ├── test_preprocessing.py  # Unit tests for text cleaning and validation routines
│   ├── test_model.py          # Unit tests for pipelines, CV, and model serialization
│   ├── test_prediction.py     # Unit tests for predictor, metrics, and schema checks
│   └── test_smoke.py          # End-to-end integration and artifact verification tests
│
└── docs/
    ├── PROJECT_REPORT.md      # Full academic college assignment report
    ├── DATASET_DOCUMENTATION.md # AG News source, citation, labels, and quality audit
    ├── METHODOLOGY.md         # Text preprocessing, TF-IDF, algorithms, and math
    ├── EVALUATION_REPORT.md   # Measured performance metrics, confusion matrix, and errors
    ├── ARCHITECTURE.md        # Modular system design and pipeline dataflow
    └── FINAL_REPORT.md        # Executive submission summary
```

---

## 3. Data Ingestion and Caching Strategy
- **Training Set (24,000 records):** Sampled via stratified random sampling (`random_state=42`) across the 4 classes to balance training speed (< 6 seconds) with high statistical representativeness (> 89.9% accuracy).
- **Test Set (7,600 records):** Uses the complete official held-out test split for zero data leakage and uncompromised evaluation.
- **In-Memory Caching:** The Streamlit frontend utilizes `@st.cache_resource` for the fitted model pipeline (instant subsequent loads) and `@st.cache_data` for static evaluation metrics and dataset explorers.
