"""Unit and integration tests for prediction engine, metrics computation, and dataset validation.

Author: Daniel Richard Xavier
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.prediction import NewsLensPredictor
from src.model import build_logistic_regression_pipeline
from src.evaluation import compute_classification_metrics
from src.data_loader import validate_dataset, get_category_mappings


@pytest.fixture
def fitted_predictor():
    """Predictor backed by a small fitted pipeline fixture."""
    texts = [
        "President signs international peace treaty at summit.",
        "Striker scores hat-trick in football tournament final.",
        "Central bank cuts interest rates to stimulate economy.",
        "Space agency launches new robotic satellite to Mars."
    ]
    labels = [0, 1, 2, 3]  # World, Sports, Business, Sci/Tech
    pipe = build_logistic_regression_pipeline(max_features=200)
    pipe.fit(pd.Series(texts), pd.Series(labels))
    return NewsLensPredictor(pipeline=pipe, metadata={"model_name": "Fixture Logistic Regression"})


def test_predictor_valid_inference(fitted_predictor):
    res = fitted_predictor.predict(text="Football team advances to cup final after penalty shootout.")
    assert res["success"] is True
    assert res["error"] is None
    assert res["predicted_category"] in ["World", "Sports", "Business", "Sci/Tech"]
    assert 0.0 <= res["confidence"] <= 1.0
    assert len(res["probabilities"]) == 4
    assert sum(res["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)


def test_predictor_headline_and_body_combination(fitted_predictor):
    h = "Wall Street Tech Stocks Surge"
    b = "Major technology corporations report higher quarterly earnings, lifting the Nasdaq index."
    res = fitted_predictor.predict(headline=h, body=b)
    assert res["success"] is True
    assert "Wall Street Tech Stocks Surge" in res["cleaned_text"]
    assert "Major technology corporations" in res["cleaned_text"]


def test_predictor_handles_empty_input(fitted_predictor):
    res = fitted_predictor.predict(text="")
    assert res["success"] is False
    assert res["predicted_category"] is None
    assert "empty or missing" in res["error"].lower()


def test_predictor_handles_whitespace_input(fitted_predictor):
    res = fitted_predictor.predict(text="   \n\t  ")
    assert res["success"] is False
    assert res["predicted_category"] is None


def test_predictor_handles_too_short_input(fitted_predictor):
    res = fitted_predictor.predict(text="Hi")
    assert res["success"] is False
    assert "too short" in res["error"].lower()


def test_compute_classification_metrics_exact():
    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    metrics = compute_classification_metrics(y_true, y_pred)

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["macro_precision"] == 1.0
    assert metrics["macro_recall"] == 1.0
    assert metrics["total_test_samples"] == 8


def test_compute_classification_metrics_with_errors():
    y_true = np.array([0, 1, 2, 3])
    y_pred = np.array([0, 1, 3, 3])  # one error (class 2 predicted as 3)
    metrics = compute_classification_metrics(y_true, y_pred)

    assert metrics["accuracy"] == 0.75
    assert metrics["macro_f1"] < 1.0


def test_validate_dataset_valid():
    df = pd.DataFrame({
        "text": ["Headline 1", "Headline 2", "Headline 3", "Headline 4"],
        "label": [0, 1, 2, 3]
    })
    summary = validate_dataset(df, "TestFixture")
    assert summary["num_rows"] == 4
    assert summary["null_text_count"] == 0
    assert summary["null_label_count"] == 0


def test_validate_dataset_missing_columns():
    df = pd.DataFrame({"headline": ["Title 1"], "category": [0]})
    with pytest.raises(ValueError, match="missing required columns"):
        validate_dataset(df)


def test_validate_dataset_invalid_labels():
    df = pd.DataFrame({
        "text": ["Article 1", "Article 2"],
        "label": [0, 99]  # 99 is invalid
    })
    with pytest.raises(ValueError, match="invalid labels"):
        validate_dataset(df)
