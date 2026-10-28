# Final Project Summary: NewsLens AI

**Project Title:** NewsLens AI — News Article Classification Using Machine Learning  
**Student & Developer:** Daniel Richard Xavier  
**Academic Module:** College Assignment Mini Project 9  
**GitHub Repository:** `https://github.com/drx2k06/miniproject9.git`  
**Execution Environment:** Python 3.12, Scikit-learn 1.9, Streamlit 1.66 on Windows (CPU)  

---

## 1. Quick Facts & Deliverables

| Deliverable | Details |
| :--- | :--- |
| **Problem Statement** | Automated news topic categorization across 4 standard domains |
| **Benchmark Dataset** | AG News Topic Classification (Zhang et al., NeurIPS 2015) |
| **Categories Supported** | 0: World, 1: Sports, 2: Business, 3: Sci/Tech |
| **Training Partition** | 24,000 stratified samples (6,000 per class, random seed 42) |
| **Held-Out Test Set** | 7,600 full benchmark articles (1,900 per class) |
| **Feature Extraction** | Sublinear TF-IDF (1-2 word n-grams, 25k vocabulary) |
| **Models Evaluated** | Multinomial Naive Bayes (Baseline) & Logistic Regression (Champion) |
| **Model Selection** | 5-Fold Stratified Cross-Validation on training split |
| **Champion Model** | Logistic Regression (L2 regularization, $C=1.0$, lbfgs) |
| **Champion Test Accuracy** | **89.97%** |
| **Champion Macro F1** | **0.8993** |
| **Fitted Model Size** | **1.01 MB** (`models/newslens_pipeline.joblib`) |
| **Automated Tests** | **36/36 passing** via Pytest |
| **Interactive UI** | Multi-page Streamlit Dashboard (`app.py`) |

---

## 2. Key Commands Quick Reference

### Environment Setup
```bash
pip install -r requirements.txt
```

### Model Training & Artifact Generation
```bash
python train.py
```

### Standalone Model Evaluation
```bash
python evaluate.py
```

### Automated Testing
```bash
python -m pytest tests/ -v
```

### Launch Interactive Web Application
```bash
streamlit run app.py
```

---

## 3. Verified Artifact Inventory

- `app.py`: Streamlit multi-page web application.
- `train.py`: Automated pipeline trainer, CV evaluator, and artifact generator.
- `evaluate.py`: Standalone test evaluation script.
- `src/data_loader.py`: AG News benchmark ingestion and schema validation.
- `src/preprocessing.py`: NLP text cleaner, HTML unescaper, length validator.
- `src/model.py`: Pipeline architectures and joblib serialization.
- `src/prediction.py`: Inference engine with multi-class probability scoring.
- `src/evaluation.py`: Metric calculation, error analysis, and plot generation.
- `models/newslens_pipeline.joblib`: 1.01 MB compressed fitted model.
- `outputs/metrics.json`: Master JSON metrics record.
- `outputs/confusion_matrix.png`: High-resolution test confusion matrix heatmap.
- `outputs/model_comparison.png`: Model comparison bar chart.
- `outputs/error_analysis.json`: Representative misclassified test cases.
- `tests/`: 36 unit, integration, and smoke tests.
- `docs/`: Complete academic reports and architectural guides.
