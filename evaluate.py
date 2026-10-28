"""Standalone evaluation script for NewsLens AI.

Loads the saved fitted pipeline, runs inference on the AG News test set,
prints the detailed classification report, confusion matrix, and exports an evaluation summary.

Author: Daniel Richard Xavier
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

from src.data_loader import load_data, get_category_mappings
from src.model import load_pipeline
from src.evaluation import compute_classification_metrics, analyze_misclassifications


def run_evaluation(
    model_path: str = "models/newslens_pipeline.joblib",
    data_dir: str = "data",
    output_dir: str = "outputs"
) -> None:
    """Run standalone model evaluation."""
    print("=" * 70)
    print("NEWLENS AI - MODEL EVALUATION SUITE")
    print("Developer: Daniel Richard Xavier")
    print("=" * 70)

    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found! Please run train.py first.")
        sys.exit(1)

    print(f"\nLoading fitted pipeline from: {model_path}...")
    pipeline, metadata = load_pipeline(model_path)
    print(f"Model Name: {metadata.get('model_name', 'Fitted Pipeline')}")
    print(f"Trained Date: {metadata.get('trained_date', 'N/A')}")
    print(f"Training Samples: {metadata.get('train_samples', 'N/A')}")

    print("\nLoading test dataset...")
    test_path = os.path.join(data_dir, "ag_news_test.parquet")
    if not os.path.exists(test_path):
        print(f"Error: Test dataset '{test_path}' not found!")
        sys.exit(1)

    test_df = pd.read_parquet(test_path)
    X_test = test_df["text"]
    y_test = test_df["label"]
    print(f"Test samples: {len(test_df):,} | Categories: 4 (World, Sports, Business, Sci/Tech)")

    id_to_name, _ = get_category_mappings()
    class_names = [id_to_name[i] for i in sorted(id_to_name.keys())]

    print("\nRunning test-set inference...")
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test) if hasattr(pipeline, "predict_proba") else None

    metrics = compute_classification_metrics(y_test, y_pred, class_names=class_names)

    print("\n" + "=" * 50)
    print("GLOBAL PERFORMANCE METRICS")
    print("=" * 50)
    print(f"  Accuracy:         {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Macro Precision:  {metrics['macro_precision']:.4f}")
    print(f"  Macro Recall:     {metrics['macro_recall']:.4f}")
    print(f"  Macro F1-Score:   {metrics['macro_f1']:.4f}")
    print(f"  Weighted F1:      {metrics['weighted_f1']:.4f}")

    print("\n" + "=" * 50)
    print("PER-CLASS CLASSIFICATION REPORT")
    print("=" * 50)
    print(f"{'Category':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 58)
    for cname in class_names:
        c_stats = metrics["per_class"][cname]
        print(f"{cname:<12} | {c_stats['precision']:<10.4f} | {c_stats['recall']:<10.4f} | {c_stats['f1_score']:<10.4f} | {c_stats['support']:<8}")

    print("\n" + "=" * 50)
    print("CONFUSION MATRIX")
    print("=" * 50)
    cm = np.array(metrics["confusion_matrix"])
    header = "True \\ Pred  " + "  ".join([f"{c[:8]:>8}" for c in class_names])
    print(header)
    print("-" * len(header))
    for i, cname in enumerate(class_names):
        row_str = f"{cname[:10]:<12}" + "  ".join([f"{cm[i, j]:>8}" for j in range(len(class_names))])
        print(row_str)

    # Save summary
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "evaluation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nEvaluation summary saved to: {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate NewsLens AI fitted model on test data.")
    parser.add_argument("--model_path", type=str, default="models/newslens_pipeline.joblib")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="outputs")
    args = parser.parse_args()

    run_evaluation(
        model_path=args.model_path,
        data_dir=args.data_dir,
        output_dir=args.output_dir
    )
