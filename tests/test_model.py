"""Unit tests for model architectures, pipeline training, and persistence.

Uses small synthetic fixtures clearly isolated from the actual AG News benchmark dataset.

Author: Daniel Richard Xavier
"""

import os
import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.model import (
    build_naive_bayes_pipeline,
    build_logistic_regression_pipeline,
    save_pipeline,
    load_pipeline,
    evaluate_cross_validation
)


@pytest.fixture
def synthetic_corpus():
    """Small synthetic news dataset for unit testing."""
    texts = [
        # World (0)
        "United Nations delegates discuss peace treaty in Geneva summit.",
        "Diplomatic talks resume between prime ministers over trade borders.",
        "Global leaders convene to address international security accords.",
        # Sports (1)
        "Manchester United scores winning goal in injury time derby.",
        "Lakers defeat Celtics in thrilling overtime basketball contest.",
        "Tennis grand slam champion reaches quarterfinal after five sets.",
        # Business (2)
        "Wall Street stocks close higher amid strong corporate earnings reports.",
        "Central bank signals possible interest rate hike next quarter.",
        "Treasury bond yields decline as retail sales and GDP data beat estimates.",
        # Sci/Tech (3)
        "Scientists discover exoplanet orbiting nearby star using James Webb telescope.",
        "Software giant releases new generative AI model with billion parameters.",
        "Cybersecurity researchers patch critical zero-day vulnerability in operating system."
    ]
    labels = [
        0, 0, 0,
        1, 1, 1,
        2, 2, 2,
        3, 3, 3
    ]
    return pd.Series(texts), pd.Series(labels)


def test_build_naive_bayes_pipeline():
    pipe = build_naive_bayes_pipeline(max_features=500)
    assert isinstance(pipe, Pipeline)
    assert "tfidf" in pipe.named_steps
    assert "classifier" in pipe.named_steps
    assert pipe.named_steps["classifier"].__class__.__name__ == "MultinomialNB"


def test_build_logistic_regression_pipeline():
    pipe = build_logistic_regression_pipeline(max_features=500)
    assert isinstance(pipe, Pipeline)
    assert "tfidf" in pipe.named_steps
    assert "classifier" in pipe.named_steps
    assert pipe.named_steps["classifier"].__class__.__name__ == "LogisticRegression"


def test_naive_bayes_fit_and_predict(synthetic_corpus):
    X, y = synthetic_corpus
    pipe = build_naive_bayes_pipeline(max_features=500)
    pipe.fit(X, y)

    query = ["Premier League team wins football championship match."]
    pred = pipe.predict(query)
    assert len(pred) == 1
    assert pred[0] in [0, 1, 2, 3]


def test_logistic_regression_fit_and_predict(synthetic_corpus):
    X, y = synthetic_corpus
    pipe = build_logistic_regression_pipeline(max_features=500)
    pipe.fit(X, y)

    query = ["Stock market indices rally following quarterly profit reports."]
    pred = pipe.predict(query)
    assert len(pred) == 1
    assert pred[0] in [0, 1, 2, 3]


def test_probability_distributions_sum_to_one(synthetic_corpus):
    X, y = synthetic_corpus
    pipe = build_logistic_regression_pipeline(max_features=500)
    pipe.fit(X, y)

    query = ["Astronomers detect radio bursts from distant galaxy."]
    probs = pipe.predict_proba(query)[0]
    assert len(probs) == 4
    assert np.isclose(np.sum(probs), 1.0, atol=1e-5)
    assert all(0.0 <= p <= 1.0 for p in probs)


def test_pipeline_serialization_and_reloading(synthetic_corpus, tmp_path):
    X, y = synthetic_corpus
    pipe = build_logistic_regression_pipeline(max_features=500)
    pipe.fit(X, y)

    save_path = str(tmp_path / "test_pipe.joblib")
    meta = {"model_name": "TestModel", "version": "1.0"}
    save_pipeline(pipe, save_path, metadata=meta)

    assert os.path.exists(save_path)

    reloaded_pipe, reloaded_meta = load_pipeline(save_path)
    assert reloaded_meta["model_name"] == "TestModel"

    pred_original = pipe.predict(["New quantum computer announced."])
    pred_reloaded = reloaded_pipe.predict(["New quantum computer announced."])
    assert pred_original[0] == pred_reloaded[0]


def test_evaluate_cross_validation(synthetic_corpus):
    X, y = synthetic_corpus
    pipe = build_naive_bayes_pipeline(max_features=500)
    res = evaluate_cross_validation(pipe, X, y, cv_folds=2, random_state=42)

    assert "cv_accuracy_mean" in res
    assert "cv_macro_f1_mean" in res
    assert 0.0 <= res["cv_accuracy_mean"] <= 1.0
