"""
Evaluation utilities: confusion matrix plots, per-class analysis,
result aggregation.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("innovation_prediction.evaluate")


def format_results_table(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Format multiple model results into a comparison table.

    Args:
        results: {model_name: metrics_dict}
    Returns:
        DataFrame with models as rows, metrics as columns
    """
    rows = []
    for model_name, metrics in results.items():
        rows.append({
            "Model": model_name,
            "Accuracy": metrics.get("accuracy", 0),
            "Macro F1": metrics.get("macro_f1", 0),
            "Weighted F1": metrics.get("weighted_f1", 0),
            "Macro Precision": metrics.get("macro_precision", 0),
            "Macro Recall": metrics.get("macro_recall", 0),
        })
    df = pd.DataFrame(rows).sort_values("Macro F1", ascending=False)
    return df


def per_class_analysis(y_true: np.ndarray, y_pred: np.ndarray,
                        y_proba: Optional[np.ndarray] = None,
                        class_names: Optional[Dict[int, str]] = None
                        ) -> pd.DataFrame:
    """
    Detailed per-class metrics including confusion patterns.

    Args:
        y_true: true labels
        y_pred: predicted labels
        y_proba: optional predicted probabilities
        class_names: optional {label: name} mapping. If None, uses Cluster_0, etc.
    """
    from sklearn.metrics import classification_report, confusion_matrix

    classes = sorted(set(y_true))
    if class_names is None:
        class_names = {i: f"Cluster_{i}" for i in classes}

    target_names = [class_names.get(i, f"Cluster_{i}") for i in classes]

    report = classification_report(
        y_true, y_pred, target_names=target_names, output_dict=True,
    )

    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]

    rows = []
    for i in range(n_classes):
        name = target_names[i]
        total = cm[i].sum()

        # Most confused with
        confusion_vec = cm[i].copy()
        confusion_vec[i] = 0
        most_confused_idx = confusion_vec.argmax()
        most_confused_name = (target_names[most_confused_idx]
                              if most_confused_idx < len(target_names)
                              else f"Cluster_{most_confused_idx}")
        most_confused_pct = confusion_vec[most_confused_idx] / max(total, 1) * 100

        rows.append({
            "Class": name,
            "Support": int(report.get(name, {}).get("support", 0)),
            "Precision": report.get(name, {}).get("precision", 0),
            "Recall": report.get(name, {}).get("recall", 0),
            "F1": report.get(name, {}).get("f1-score", 0),
            "Most Confused With": most_confused_name,
            "Confusion %": most_confused_pct,
        })

    return pd.DataFrame(rows)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                           save_path: Optional[str] = None,
                           class_names: Optional[Dict[int, str]] = None):
    """Generate and optionally save confusion matrix plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true, y_pred)
        n = cm.shape[0]
        if class_names is None:
            labels = [f"Cluster_{i}" for i in range(n)]
        else:
            labels = [class_names.get(i, f"Cluster_{i}") for i in range(n)]

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                     xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Confusion matrix saved to {save_path}")
        plt.close(fig)
    except ImportError:
        logger.warning("matplotlib/seaborn not available for plotting")


def plot_training_curves(train_losses: List[float],
                          val_metrics: List[Dict],
                          save_path: Optional[str] = None):
    """Plot training loss and validation metrics over epochs."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = range(1, len(train_losses) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Training loss
        ax1.plot(epochs, train_losses, "b-", label="Train Loss")
        val_losses = [m.get("eval_loss", 0) for m in val_metrics]
        ax1.plot(epochs[:len(val_losses)], val_losses, "r-", label="Val Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training & Validation Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Validation metrics
        val_f1 = [m.get("macro_f1", 0) for m in val_metrics]
        val_acc = [m.get("accuracy", 0) for m in val_metrics]
        ax2.plot(epochs[:len(val_f1)], val_f1, "g-", label="Macro F1")
        ax2.plot(epochs[:len(val_acc)], val_acc, "b-", label="Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Score")
        ax2.set_title("Validation Metrics")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            logger.info(f"Training curves saved to {save_path}")
        plt.close(fig)
    except ImportError:
        logger.warning("matplotlib not available for plotting")
