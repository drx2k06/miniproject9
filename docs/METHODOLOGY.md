# Methodology: NewsLens AI Classification Pipeline

## 1. Problem Formulation
Given an arbitrary news headline $h$ and description text $d$, text classification maps the combined sequence $x = (h, d)$ to one of $K = 4$ discrete topic categories $y \in \{0, 1, 2, 3\}$:
$$f: \mathcal{X} \to \mathcal{Y}$$
where $\mathcal{Y} = \{\text{World}, \text{Sports}, \text{Business}, \text{Sci/Tech}\}$.

## 2. Text Preprocessing Pipeline
To preserve domain entities while eliminating web feed noise, input text undergoes sequential normalization in `src/preprocessing.py`:
1. **Type & Null Validation:** Null, non-string, or empty entries are trapped immediately.
2. **Entity Unescaping:** HTML entities are converted to canonical UTF-8 characters (`&amp;` $\to$ `&`, `&quot;` $\to$ `"`).
3. **HTML & URL Stripping:** Residual HTML tags (`<p>`, `<b>`) and web addresses (`http://...`) are excised using regular expressions.
4. **Control Character Cleansing:** Control bytes and feed escape artifacts (`\b`, `\r`, `\t`) are replaced with standard spaces.
5. **Whitespace Collapsing:** Consecutive spaces, tabs, and newlines are collapsed into a single whitespace separator, followed by string stripping.

## 3. Feature Extraction: Term Frequency - Inverse Document Frequency (TF-IDF)
Raw text tokens are transformed into numerical vector space using TF-IDF representation.

### Formulation
For term $t$ in document $d$ within corpus $D$:
$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

1. **Sublinear Term Frequency Scaling:**
   $$\text{TF}(t, d) = 1 + \log(f_{t, d}) \quad \text{for } f_{t, d} > 0$$
   *Rationale:* A word appearing 20 times in an article is informative, but not 20 times more informative than a word appearing once. Logarithmic dampening prevents vocabulary frequency distortion.

2. **Inverse Document Frequency with Smooth Weighting:**
   $$\text{IDF}(t, D) = \log\left(\frac{1 + |D|}{1 + |\{d \in D : t \in d\}|}\right) + 1$$
   Terms that occur across almost every document (common stop words) receive low weights, while domain-discriminative terms receive high weights.

3. **N-Gram Range & Vocabulary Pruning:**
   - Word unigrams and bigrams: $n \in \{1, 2\}$.
   - Maximum vocabulary size: 25,000 top features.
   - Stop words: Standard English stop word elimination.
   - Token pattern: `(?u)\b[a-zA-Z0-9_\-\.]{2,}\b` (preserves hyphens, dots, and numbers like `iPhone-15`, `.NET`, `3.5%`).

4. **L2 Unit Normalization:**
   Each document vector $\mathbf{v}$ is normalized such that $\|\mathbf{v}\|_2 = 1$, making representations invariant to document length.

## 4. Candidate Machine Learning Models

### Model 1: Multinomial Naive Bayes (Baseline)
- **Model Type:** Generative probabilistic classifier.
- **Formulation:**
  $$P(y = c \mid \mathbf{x}) \propto P(y = c) \prod_{j=1}^{V} P(x_j \mid y = c)^{x_j}$$
- **Feature Probability with Laplace Smoothing ($\alpha = 0.1$):**
  $$\hat{\theta}_{c, j} = \frac{N_{c, j} + \alpha}{N_c + \alpha V}$$
  where $N_{c, j}$ is the sum of TF-IDF feature weights for term $j$ in class $c$, $N_c$ is total term weight across class $c$, and $V = 25,000$ is vocabulary size.
- **Characteristics:** Extremely fast ($O(N \cdot d)$ training time), low memory footprint, serves as the standard academic baseline.

### Model 2: Logistic Regression (Champion)
- **Model Type:** Discriminative linear classifier.
- **Formulation:** For multi-class classification with $K = 4$ classes, the model fits weight vectors $\mathbf{w}_c$ and bias $b_c$ for each category $c \in \{0, 1, 2, 3\}$.
- **Softmax Probability:**
  $$P(y = c \mid \mathbf{x}) = \frac{\exp(\mathbf{w}_c^T \mathbf{x} + b_c)}{\sum_{k=0}^{K-1} \exp(\mathbf{w}_k^T \mathbf{x} + b_k)}$$
- **Optimization Objective:** Minimizes penalized categorical cross-entropy loss with L2 regularization:
  $$\min_{\mathbf{w}} \left\{ - \sum_{i=1}^{N} \sum_{k=0}^{K-1} \mathbf{1}\{y_i = k\} \log P(y = k \mid \mathbf{x}_i) + \frac{1}{2C} \sum_{k=0}^{K-1} \|\mathbf{w}_k\|_2^2 \right\}$$
- **Hyperparameters:** Inverse regularization $C = 1.0$, solver = `lbfgs` (Limited-memory Broyden–Fletcher–Goldfarb–Shanno), `max_iter = 500`.

## 5. Model Selection & Data Leakage Prevention Strategy
To guarantee scientific validity:
1. **Pipeline Encapsulation:** The preprocessing function, `TfidfVectorizer`, and the classifier are combined into an atomic `sklearn.pipeline.Pipeline`.
2. **K-Fold Stratified Cross-Validation:** Hyperparameters and candidate comparisons are conducted strictly on the training partition using 5-fold Stratified CV.
3. **Pristine Test Split:** The 7,600-sample test set is withheld completely during vectorizer fitting and model tuning. It is evaluated only once on the final fitted pipelines.
