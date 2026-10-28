"""Model evaluation, metrics computation, error analysis, and visualization for NewsLens AI.

Author: Daniel Richard Xavier
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from src.data_loader import CATEGORY_MAP_ID_TO_NAME


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Compute comprehensive classification evaluation metrics.
    
    Args:
        y_true: Ground truth labels (integer array or list).
        y_pred: Predicted labels.
        class_names: Optional list of class names in ascending ID order.

    Returns:
        Structured dictionary of overall and per-class metrics.
    """
    if class_names is None:
        class_names = [CATEGORY_MAP_ID_TO_NAME[i] for i in sorted(CATEGORY_MAP_ID_TO_NAME.keys())]

    acc = float(accuracy_score(y_true, y_pred))
    macro_prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    cm_list = cm.tolist()

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    per_class = {}
    for idx, cname in enumerate(class_names):
        if cname in report_dict:
            per_class[cname] = {
                "class_id": idx,
                "precision": float(round(report_dict[cname]["precision"], 4)),
                "recall": float(round(report_dict[cname]["recall"], 4)),
                "f1_score": float(round(report_dict[cname]["f1-score"], 4)),
                "support": int(report_dict[cname]["support"])
            }

    metrics = {
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "confusion_matrix": cm_list,
        "per_class": per_class,
        "classes": class_names,
        "total_test_samples": len(y_true)
    }
    return metrics


def generate_confusion_matrix_plot(
    cm: np.ndarray,
    class_names: List[str],
    output_path: str,
    title: str = "NewsLens AI - Test Set Confusion Matrix"
) -> None:
    """Generate and save a publication-quality confusion matrix heatmap.
    
    Args:
        cm: 2D numpy array confusion matrix.
        class_names: List of class labels for axes.
        output_path: Filepath where plot image will be saved.
        title: Title of the chart.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(8, 6), dpi=300)

    # Compute row percentages for dual annotation (count + row %)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    annot_matrix = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot_matrix[i, j] = f"{cm[i, j]:,}\n({cm_normalized[i, j]:.1%})"

    sns.heatmap(
        cm,
        annot=annot_matrix,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        linewidths=1.0,
        linecolor="#e0e0e0"
    )

    plt.title(title, fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Predicted Category", fontsize=12, fontweight="semibold", labelpad=10)
    plt.ylabel("True Category", fontsize=12, fontweight="semibold", labelpad=10)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def generate_model_comparison_plot(
    metrics_comparison: Dict[str, Dict[str, float]],
    output_path: str
) -> None:
    """Generate and save a side-by-side bar chart comparing model performances.
    
    Args:
        metrics_comparison: Dict of model_name -> {metric_name: value}.
        output_path: Filepath where plot image will be saved.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    metric_keys = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"]
    display_names = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1", "Weighted F1"]

    models = list(metrics_comparison.keys())
    x = np.arange(len(metric_keys))
    width = 0.35

    plt.figure(figsize=(10, 6), dpi=300)
    colors = ["#3498db", "#2ecc71"]

    for idx, model in enumerate(models):
        values = [metrics_comparison[model].get(k, 0.0) for k in metric_keys]
        bars = plt.bar(x + (idx - 0.5) * width, values, width, label=model, color=colors[idx % len(colors)], alpha=0.9)
        # Add value labels above bars
        for bar in bars:
            height = bar.get_height()
            plt.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="semibold"
            )

    plt.title("Model Performance Comparison on Unseen AG News Test Set", fontsize=14, fontweight="bold", pad=15)
    plt.ylabel("Score (0.0 to 1.0)", fontsize=12, fontweight="semibold")
    plt.xticks(x, display_names, fontsize=11, fontweight="semibold")
    plt.ylim(0.70, 1.0)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend(loc="lower right", frameon=True, shadow=True, fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def analyze_misclassifications(
    texts: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    n_samples: int = 15
) -> List[Dict[str, Any]]:
    """Extract representative misclassified examples with confidence and rationale.
    
    Args:
        texts: Raw text inputs.
        y_true: True integer labels.
        y_pred: Predicted integer labels.
        y_prob: Predicted probability array.
        class_names: Class labels.
        n_samples: Number of misclassified examples to return.

    Returns:
        List of misclassified item dictionaries.
    """
    if class_names is None:
        class_names = [CATEGORY_MAP_ID_TO_NAME[i] for i in sorted(CATEGORY_MAP_ID_TO_NAME.keys())]

    misclassified_mask = (y_true != y_pred)
    indices = np.where(misclassified_mask)[0]

    examples = []
    for idx in indices[:n_samples]:
        true_id = int(y_true[idx])
        pred_id = int(y_pred[idx])
        conf = float(round(y_prob[idx][pred_id], 4)) if y_prob is not None else 0.0

        item = {
            "index": int(idx),
            "true_label": class_names[true_id],
            "true_label_id": true_id,
            "predicted_label": class_names[pred_id],
            "predicted_label_id": pred_id,
            "confidence": conf,
            "text": str(texts.iloc[idx]),
            "snippet": str(texts.iloc[idx])[:150] + ("..." if len(str(texts.iloc[idx])) > 150 else "")
        }
        examples.append(item)

    return examples
