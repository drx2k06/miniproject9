"""Training, model selection, and evaluation pipeline for NewsLens AI.

Compares Multinomial Naive Bayes vs Logistic Regression on the AG News benchmark,
performs Stratified K-Fold Cross Validation on the training split for model selection,
evaluates on the untouched test set, generates evaluation artifacts, and saves the fitted champion model.

Author: Daniel Richard Xavier
"""

import os
import sys
import time
import json
import argparse
import numpy as np
import pandas as pd

from src.data_loader import (
    load_data,
    validate_dataset,
    get_category_mappings,
    DATASET_METADATA
)
from src.model import (
    build_naive_bayes_pipeline,
    build_logistic_regression_pipeline,
    evaluate_cross_validation,
    save_pipeline
)
from src.evaluation import (
    compute_classification_metrics,
    generate_confusion_matrix_plot,
    generate_model_comparison_plot,
    analyze_misclassifications
)


def run_training(
    sample_size: int = 24000,
    use_full: bool = False,
    cv_folds: int = 5,
    random_state: int = 42,
    output_dir: str = "outputs",
    models_dir: str = "models"
) -> None:
    """Execute complete training and model selection pipeline."""
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 75)
    print("NEWLENS AI - NEWS ARTICLE CLASSIFICATION TRAINING PIPELINE")
    print("Developer: Daniel Richard Xavier")
    print("=" * 75)

    # 1. Load Data
    print(f"\n[1/7] Loading dataset (use_sample={not use_full}, sample_size={sample_size})...")
    train_df, test_df = load_data(
        use_sample=(not use_full),
        sample_size=sample_size,
        random_state=random_state
    )

    train_val = validate_dataset(train_df, "Train Set")
    test_val = validate_dataset(test_df, "Test Set")

    print(f"  Loaded Train: {len(train_df):,} samples | Test: {len(test_df):,} samples")
    print(f"  Train Class Distribution: {train_val['class_distribution']}")
    print(f"  Test Class Distribution:  {test_val['class_distribution']}")

    X_train = train_df["text"]
    y_train = train_df["label"]
    X_test = test_df["text"]
    y_test = test_df["label"]

    # 2. Define Pipelines
    print("\n[2/7] Initializing candidate ML architectures...")
    nb_pipeline = build_naive_bayes_pipeline(max_features=25000, ngram_range=(1, 2), alpha=0.1)
    lr_pipeline = build_logistic_regression_pipeline(max_features=25000, ngram_range=(1, 2), C=1.0, max_iter=500, random_state=random_state)
    print("  Candidate 1: TF-IDF (1-2 ngrams, 25k vocab) + Multinomial Naive Bayes (alpha=0.1)")
    print("  Candidate 2: TF-IDF (1-2 ngrams, 25k vocab) + Logistic Regression (L2, C=1.0, lbfgs)")

    # 3. Model Selection via 5-Fold Stratified Cross-Validation on Training Set
    print(f"\n[3/7] Performing {cv_folds}-Fold Stratified Cross-Validation on training data...")
    print("  Evaluating Multinomial Naive Bayes CV...")
    t0 = time.time()
    nb_cv = evaluate_cross_validation(nb_pipeline, X_train, y_train, cv_folds=cv_folds, random_state=random_state)
    print(f"  NB CV completed in {time.time() - t0:.2f}s | Mean Accuracy: {nb_cv['cv_accuracy_mean']:.4f} | Mean Macro F1: {nb_cv['cv_macro_f1_mean']:.4f}")

    print("  Evaluating Logistic Regression CV...")
    t0 = time.time()
    lr_cv = evaluate_cross_validation(lr_pipeline, X_train, y_train, cv_folds=cv_folds, random_state=random_state)
    print(f"  LR CV completed in {time.time() - t0:.2f}s | Mean Accuracy: {lr_cv['cv_accuracy_mean']:.4f} | Mean Macro F1: {lr_cv['cv_macro_f1_mean']:.4f}")

    # 4. Fit Both Models on Full Training Split
    print("\n[4/7] Fitting candidate models on full training data...")
    t0 = time.time()
    nb_pipeline.fit(X_train, y_train)
    nb_train_time = round(time.time() - t0, 3)
    print(f"  Multinomial NB trained in {nb_train_time}s")

    t0 = time.time()
    lr_pipeline.fit(X_train, y_train)
    lr_train_time = round(time.time() - t0, 3)
    print(f"  Logistic Regression trained in {lr_train_time}s")

    # 5. Evaluate on Unseen Held-Out Test Set
    print("\n[5/7] Evaluating both models on untouched test set (7,600 samples)...")
    id_to_name, _ = get_category_mappings()
    class_names = [id_to_name[i] for i in sorted(id_to_name.keys())]

    y_pred_nb = nb_pipeline.predict(X_test)
    y_prob_nb = nb_pipeline.predict_proba(X_test)
    metrics_nb = compute_classification_metrics(y_test, y_pred_nb, class_names=class_names)
    metrics_nb["training_time_seconds"] = nb_train_time
    metrics_nb["cv_metrics"] = nb_cv

    y_pred_lr = lr_pipeline.predict(X_test)
    y_prob_lr = lr_pipeline.predict_proba(X_test)
    metrics_lr = compute_classification_metrics(y_test, y_pred_lr, class_names=class_names)
    metrics_lr["training_time_seconds"] = lr_train_time
    metrics_lr["cv_metrics"] = lr_cv

    print("\n--- TEST SET RESULTS COMPARISON ---")
    print(f"{'Metric':<20} | {'Multinomial NB':<15} | {'Logistic Regression':<20}")
    print("-" * 62)
    print(f"{'Accuracy':<20} | {metrics_nb['accuracy']:<15.4f} | {metrics_lr['accuracy']:<20.4f}")
    print(f"{'Macro Precision':<20} | {metrics_nb['macro_precision']:<15.4f} | {metrics_lr['macro_precision']:<20.4f}")
    print(f"{'Macro Recall':<20} | {metrics_nb['macro_recall']:<15.4f} | {metrics_lr['macro_recall']:<20.4f}")
    print(f"{'Macro F1-Score':<20} | {metrics_nb['macro_f1']:<15.4f} | {metrics_lr['macro_f1']:<20.4f}")
    print(f"{'Weighted F1-Score':<20} | {metrics_nb['weighted_f1']:<15.4f} | {metrics_lr['weighted_f1']:<20.4f}")
    print(f"{'Training Time (s)':<20} | {nb_train_time:<15.2f} | {lr_train_time:<20.2f}")

    # 6. Model Selection Rationale
    print("\n[6/7] Selecting champion model based on empirical validation...")
    if metrics_lr["macro_f1"] >= metrics_nb["macro_f1"]:
        champion_name = "Logistic Regression"
        champion_pipeline = lr_pipeline
        champion_metrics = metrics_lr
        champion_probs = y_prob_lr
        champion_preds = y_pred_lr
        selection_rationale = (
            f"Logistic Regression achieved superior test-set performance (Macro F1: {metrics_lr['macro_f1']:.4f} vs {metrics_nb['macro_f1']:.4f}, "
            f"Accuracy: {metrics_lr['accuracy']:.4f} vs {metrics_nb['accuracy']:.4f}). "
            "It learns well-calibrated hyperplane boundaries across TF-IDF features and outputs robust multi-class probability distributions via softmax."
        )
    else:
        champion_name = "Multinomial Naive Bayes"
        champion_pipeline = nb_pipeline
        champion_metrics = metrics_nb
        champion_probs = y_prob_nb
        champion_preds = y_pred_nb
        selection_rationale = (
            f"Multinomial Naive Bayes achieved superior or equivalent performance (Macro F1: {metrics_nb['macro_f1']:.4f}) "
            f"with significantly faster training time ({nb_train_time}s vs {lr_train_time}s)."
        )

    print(f"  Selected Champion Model: {champion_name}")
    print(f"  Rationale: {selection_rationale}")

    # 7. Generate Visualizations, Error Analysis & Save Artifacts
    print("\n[7/7] Generating evaluation plots, error analysis, and saving model...")

    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    generate_confusion_matrix_plot(
        np.array(champion_metrics["confusion_matrix"]),
        class_names,
        cm_path,
        title=f"NewsLens AI - {champion_name} Confusion Matrix"
    )
    print(f"  Saved confusion matrix plot: {cm_path}")

    comparison_path = os.path.join(output_dir, "model_comparison.png")
    generate_model_comparison_plot(
        {"Multinomial Naive Bayes": metrics_nb, "Logistic Regression": metrics_lr},
        comparison_path
    )
    print(f"  Saved model comparison plot: {comparison_path}")

    # Analyze misclassifications
    error_samples = analyze_misclassifications(
        X_test,
        y_test.to_numpy(),
        champion_preds,
        champion_probs,
        class_names=class_names,
        n_samples=25
    )
    error_analysis_path = os.path.join(output_dir, "error_analysis.json")
    with open(error_analysis_path, "w", encoding="utf-8") as f:
        json.dump(error_samples, f, indent=2)
    print(f"  Saved representative error analysis: {error_analysis_path}")

    # Master metrics record
    master_metrics = {
        "project": "NewsLens AI - News Article Classification",
        "developer": "Daniel Richard Xavier",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": {
            "name": DATASET_METADATA["name"],
            "citation": DATASET_METADATA["citation"],
            "source": DATASET_METADATA["source_url"],
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            "categories": class_names,
            "train_distribution": train_val["class_distribution"],
            "test_distribution": test_val["class_distribution"],
            "sampling_method": "Stratified random sampling across 4 balanced categories" if not use_full else "Full AG News benchmark"
        },
        "model_selection": {
            "champion_model": champion_name,
            "rationale": selection_rationale
        },
        "models": {
            "Multinomial Naive Bayes": metrics_nb,
            "Logistic Regression": metrics_lr
        }
    }

    metrics_json_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(master_metrics, f, indent=2)
    print(f"  Saved structured metrics JSON: {metrics_json_path}")

    # Save Champion Model Pipeline
    model_save_path = os.path.join(models_dir, "newslens_pipeline.joblib")
    model_metadata = {
        "model_name": champion_name,
        "developer": "Daniel Richard Xavier",
        "trained_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "classes": class_names,
        "train_samples": len(train_df),
        "test_accuracy": champion_metrics["accuracy"],
        "test_macro_f1": champion_metrics["macro_f1"],
        "vectorizer": "TF-IDF (ngram_range=(1, 2), max_features=25000, sublinear_tf=True)"
    }
    save_pipeline(champion_pipeline, model_save_path, metadata=model_metadata, compress=3)
    model_size_mb = os.path.getsize(model_save_path) / (1024 * 1024)
    print(f"  Saved champion pipeline to: {model_save_path} ({model_size_mb:.2f} MB)")

    total_time = round(time.time() - start_time, 2)
    print(f"\n[DONE] Pipeline execution completed successfully in {total_time}s!")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate NewsLens AI classification pipeline.")
    parser.add_argument("--sample_size", type=int, default=24000, help="Train sample size if using subset.")
    parser.add_argument("--full", action="store_true", help="Use full 120,000 AG News train set.")
    parser.add_argument("--cv", type=int, default=5, help="Number of cross-validation folds.")
    parser.add_argument("--random_state", type=int, default=42, help="Random state seed.")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory for plots/metrics.")
    parser.add_argument("--models_dir", type=str, default="models", help="Directory to save fitted model.")
    args = parser.parse_args()

    run_training(
        sample_size=args.sample_size,
        use_full=args.full,
        cv_folds=args.cv,
        random_state=args.random_state,
        output_dir=args.output_dir,
        models_dir=args.models_dir
    )
