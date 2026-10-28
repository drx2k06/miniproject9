"""Model building, training, cross-validation, and persistence for NewsLens AI.

Implements scikit-learn Pipelines combining TF-IDF vectorization with:
  1. Multinomial Naive Bayes (fast baseline, generative probabilistic model)
  2. Logistic Regression (linear discriminative classifier, calibrated probabilities)

Author: Daniel Richard Xavier
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.preprocessing import clean_text


def build_naive_bayes_pipeline(
    max_features: int = 25000,
    ngram_range: Tuple[int, int] = (1, 2),
    alpha: float = 0.1,
    sublinear_tf: bool = True
) -> Pipeline:
    """Build a TF-IDF + Multinomial Naive Bayes classification pipeline.
    
    Args:
        max_features: Maximum vocabulary size for TF-IDF.
        ngram_range: Lower and upper boundary of word n-grams (1, 2) captures unigrams and bigrams.
        alpha: Additive (Laplace/Lidstone) smoothing parameter.
        sublinear_tf: Apply sublinear tf scaling (1 + log(tf)).

    Returns:
        Configured scikit-learn Pipeline.
    """
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=clean_text,
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english",
            sublinear_tf=sublinear_tf,
            token_pattern=r"(?u)\b[a-zA-Z0-9_\-\.]{2,}\b"
        )),
        ("classifier", MultinomialNB(alpha=alpha))
    ])
    return pipeline


def build_logistic_regression_pipeline(
    max_features: int = 25000,
    ngram_range: Tuple[int, int] = (1, 2),
    C: float = 1.0,
    max_iter: int = 500,
    random_state: int = 42,
    sublinear_tf: bool = True
) -> Pipeline:
    """Build a TF-IDF + Logistic Regression classification pipeline.
    
    Args:
        max_features: Maximum vocabulary size for TF-IDF.
        ngram_range: Word n-gram range.
        C: Inverse regularization strength; smaller values specify stronger regularization.
        max_iter: Maximum iterations for solver convergence.
        random_state: Random state seed.
        sublinear_tf: Apply sublinear tf scaling.

    Returns:
        Configured scikit-learn Pipeline.
    """
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=clean_text,
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english",
            sublinear_tf=sublinear_tf,
            token_pattern=r"(?u)\b[a-zA-Z0-9_\-\.]{2,}\b"
        )),
        ("classifier", LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=random_state,
            solver="lbfgs"
        ))
    ])
    return pipeline


def evaluate_cross_validation(
    pipeline: Pipeline,
    X: pd.Series,
    y: pd.Series,
    cv_folds: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """Perform Stratified K-Fold cross-validation on training data.
    
    Args:
        pipeline: scikit-learn Pipeline.
        X: Feature text series.
        y: Target label series.
        cv_folds: Number of folds (default: 5).
        random_state: Random seed for fold splitting.

    Returns:
        Dictionary of mean and std cross-validation metrics.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scoring = {
        "accuracy": "accuracy",
        "macro_f1": "f1_macro",
        "weighted_f1": "f1_weighted",
        "macro_precision": "precision_macro",
        "macro_recall": "recall_macro"
    }
    scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False)

    results = {
        "cv_accuracy_mean": float(round(np.mean(scores["test_accuracy"]), 4)),
        "cv_accuracy_std": float(round(np.std(scores["test_accuracy"]), 4)),
        "cv_macro_f1_mean": float(round(np.mean(scores["test_macro_f1"]), 4)),
        "cv_macro_f1_std": float(round(np.std(scores["test_macro_f1"]), 4)),
        "cv_weighted_f1_mean": float(round(np.mean(scores["test_weighted_f1"]), 4)),
        "cv_weighted_f1_std": float(round(np.std(scores["test_weighted_f1"]), 4)),
        "cv_macro_precision_mean": float(round(np.mean(scores["test_macro_precision"]), 4)),
        "cv_macro_recall_mean": float(round(np.mean(scores["test_macro_recall"]), 4)),
        "folds": cv_folds
    }
    return results


def save_pipeline(
    pipeline: Pipeline,
    filepath: str,
    metadata: Optional[Dict[str, Any]] = None,
    compress: int = 3
) -> None:
    """Save fitted pipeline artifact to disk using joblib.
    
    Args:
        pipeline: Fitted scikit-learn pipeline.
        filepath: Target file path.
        metadata: Optional metadata dictionary to store alongside the model.
        compress: Joblib compression level (1-9).
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    payload = {
        "pipeline": pipeline,
        "metadata": metadata or {}
    }
    joblib.dump(payload, filepath, compress=compress)


def load_pipeline(filepath: str) -> Tuple[Pipeline, Dict[str, Any]]:
    """Load fitted pipeline artifact from disk.
    
    Args:
        filepath: Path to saved joblib file.

    Returns:
        Tuple of (pipeline, metadata).

    Raises:
        FileNotFoundError: If model file does not exist.
        ValueError: If file content is not a valid model payload.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model artifact not found at: {filepath}")

    payload = joblib.load(filepath)
    if isinstance(payload, dict) and "pipeline" in payload:
        return payload["pipeline"], payload.get("metadata", {})
    elif isinstance(payload, Pipeline):
        return payload, {}
    else:
        raise ValueError(f"Unexpected payload format loaded from {filepath}")
