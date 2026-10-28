"""End-to-end integration and smoke tests for NewsLens AI.

Author: Daniel Richard Xavier
"""

import os
import json
import pytest
from src.prediction import NewsLensPredictor


def test_saved_model_artifact_exists():
    model_path = os.path.join("models", "newslens_pipeline.joblib")
    assert os.path.exists(model_path), f"Saved model missing at {model_path}"
    assert os.path.getsize(model_path) > 10000, "Model file is suspiciously small"


def test_saved_model_end_to_end_predictions():
    model_path = os.path.join("models", "newslens_pipeline.joblib")
    predictor = NewsLensPredictor(model_path=model_path)

    test_samples = [
        ("Prime Minister signs international mutual defense treaty in Brussels.", "World"),
        ("Forward scores hat-trick as football club advances to championship final.", "Sports"),
        ("Central bank raises benchmark interest rate amid inflation and bond market movements.", "Business"),
        ("Scientists develop silicon photonic processor with billion transistors.", "Sci/Tech")
    ]

    for text, expected in test_samples:
        res = predictor.predict(text=text)
        assert res["success"] is True
        assert res["predicted_category"] in ["World", "Sports", "Business", "Sci/Tech"]
        assert 0.0 <= res["confidence"] <= 1.0
        assert sum(res["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)


def test_evaluation_outputs_exist():
    assert os.path.exists("outputs/metrics.json")
    assert os.path.exists("outputs/confusion_matrix.png")
    assert os.path.exists("outputs/model_comparison.png")
    assert os.path.exists("outputs/error_analysis.json")
    assert os.path.exists("outputs/evaluation_summary.json")

    with open("outputs/metrics.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        assert "models" in data
        assert "Logistic Regression" in data["models"]
        assert "Multinomial Naive Bayes" in data["models"]
        lr = data["models"]["Logistic Regression"]
        assert lr["accuracy"] > 0.85
        assert lr["macro_f1"] > 0.85


def test_app_script_exists_and_readable():
    assert os.path.exists("app.py")
    with open("app.py", "r", encoding="utf-8") as f:
        content = f.read()
        assert "NewsLens AI" in content
        assert "Daniel Richard Xavier" in content
