# Evaluation Report: NewsLens AI Benchmark Performance

**Author:** Daniel Richard Xavier  
**Dataset:** AG News Official Benchmark (7,600 Unseen Held-Out Test Articles)  
**Evaluation Date:** October 28, 2026  

---

## 1. Executive Summary
NewsLens AI systematically evaluated two supervised text classification architectures across 24,000 training records and 7,600 unseen test records. Both models demonstrated competitive performance exceeding 89.8% test accuracy.

**Logistic Regression** was selected as the **Champion Model** based on superior Macro F1-score (**0.8993**), overall test accuracy (**0.8997**), and robust probability calibration via softmax output.

---

## 2. Global Test Performance Metrics

| Evaluation Metric | Multinomial Naive Bayes | Logistic Regression (Champion) | Performance Delta |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 0.8986 (89.86%) | **0.8997 (89.97%)** | **+0.11%** |
| **Macro-Averaged Precision** | 0.8983 | **0.8992** | **+0.0009** |
| **Macro-Averaged Recall** | 0.8986 | **0.8997** | **+0.0011** |
| **Macro-Averaged F1-Score** | 0.8982 | **0.8993** | **+0.0011** |
| **Weighted F1-Score** | 0.8982 | **0.8993** | **+0.0011** |
| **5-Fold CV Macro F1 (Train Split)** | 0.9009 (±0.0028) | **0.9028 (±0.0023)** | **+0.0019** |
| **Training Time (24,000 samples)** | **4.23 seconds** | 5.62 seconds | -1.39s |
| **Per-Article Inference Latency** | < 2 ms | < 3 ms | Real-time |

---

## 3. Category-Wise Performance Breakdown (Champion Model)

| Category Index | Category Name | Precision | Recall | F1-Score | Support (Actual Count) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **0** | **World** | 0.9109 | 0.9037 | 0.9073 | 1,900 |
| **1** | **Sports** | **0.9406** | **0.9747** | **0.9574** | 1,900 |
| **2** | **Business** | 0.8753 | 0.8500 | 0.8625 | 1,900 |
| **3** | **Sci/Tech** | 0.8701 | 0.8705 | 0.8703 | 1,900 |
| **-** | **Macro Average** | **0.8992** | **0.8997** | **0.8993** | **7,600** |
| **-** | **Weighted Average** | **0.8992** | **0.8997** | **0.8993** | **7,600** |

### Category Performance Observations
- **Highest Performing Domain — Sports (F1: 0.9574):** Sports articles possess highly distinct vocabulary (`quarterback`, `championship`, `coach`, `olympic`, `tournament`, `derby`) with minimal semantic overlap with foreign diplomacy or corporate balance sheets.
- **Strong Performance — World (F1: 0.9073):** International diplomacy terms (`peace`, `treaty`, `minister`, `ambassador`, `sanctions`) are readily identified.
- **Challenging Boundaries — Business (F1: 0.8625) vs. Sci/Tech (F1: 0.8703):** The highest rate of cross-category confusion occurred between corporate business news and technology reports.

---

## 4. Confusion Matrix Analysis

The confusion matrix on the 7,600 test articles:

```
                  Predicted World    Predicted Sports    Predicted Business    Predicted Sci/Tech    Total
True World             1,717                63                  73                    47             1,900
True Sports               22             1,852                  16                    10             1,900
True Business             71                24               1,615                   190             1,900
True Sci/Tech             75                30                 141                 1,654             1,900
Total Predicted        1,885             1,969               1,845                 1,901             7,600
```

### Detailed Matrix Observations
1. **Sports Precision & Recall:** Out of 1,900 actual sports articles, 1,852 were correctly categorized (97.47% recall).
2. **Business $\to$ Sci/Tech Confusion (190 cases):** Articles detailing major technology earnings, patent litigation, and corporate mergers involving tech conglomerates (e.g. Google, Microsoft, Intel, Oracle) were predicted as Sci/Tech.
3. **Sci/Tech $\to$ Business Confusion (141 cases):** Articles introducing new software products, hardware launches, or market commercialization of technical discoveries were predicted as Business due to financial terms like `commercial`, `revenue`, and `market launch`.

---

## 5. Error Analysis and Root Causes

Representative misclassified test articles were extracted and analyzed:

| True Category | Predicted Category | Confidence | Representative Text Snippet | Linguistic Rationale |
| :--- | :--- | :---: | :--- | :--- |
| **Business** | **Sci/Tech** | 0.8841 | *"Microsoft to offer anti-spyware software beta next month as security acquisitions integrate into core OS..."* | Dominance of technical keywords (`anti-spyware`, `software`, `beta`, `OS`) overshadows the business entity. |
| **Sci/Tech** | **Business** | 0.7915 | *"Intel cuts quarterly revenue targets citing inventory adjustments and reduced European demand for processors..."* | Heavy financial nomenclature (`revenue targets`, `inventory`, `European demand`) drives classifier to Business. |
| **World** | **Business** | 0.6842 | *"European Union imposes retaliatory trade tariffs on steel imports following bilateral dispute..."* | Trade diplomacy overlaps strongly with corporate economics. |
| **World** | **Sports** | 0.7120 | *"Olympic host city prepares heavy security perimeter and diplomatic protocol for opening ceremony..."* | Co-occurrence of `Olympic` and `security perimeter`. |

---

## 6. Model Selection Rationale
1. **Empirical Superiority:** Logistic Regression demonstrated superior accuracy and Macro F1 across both 5-fold cross-validation and the held-out test set.
2. **Probability Calibration:** Naive Bayes computes extreme posterior probabilities (often 0.999 or 0.001) due to multiplying thousands of conditional probabilities. Logistic Regression outputs well-behaved, smooth softmax probabilities.
3. **L2 Regularization:** Mitigates overfitting on co-occurring wire tokens without discarding informative feature interactions.
