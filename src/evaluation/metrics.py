"""Evaluation utilities: metrics, tables, plots, error analysis."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


# ──────────────────────────────────────────────────────────────────────────────
# Compute metrics
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: list[int],
    y_pred: list[int],
    y_prob: list[float] | None = None,
) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="binary", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="binary", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="binary", zero_division=0),
    }
    if y_prob is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        except Exception:
            metrics["roc_auc"] = float("nan")
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Performance table
# ──────────────────────────────────────────────────────────────────────────────

def build_performance_table(results: dict[str, dict]) -> pd.DataFrame:
    """Build a formatted performance table from multiple experiment results."""
    rows = []
    for exp_name, metrics in results.items():
        row = {"experiment": exp_name}
        row.update(metrics)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("experiment")
    return df


def print_performance_table(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("PERFORMANCE TABLE")
    print("=" * 60)
    print(df.round(4).to_string())
    print("=" * 60 + "\n")


def save_performance_table(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    print(f"  Performance table → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────────────────────

def plot_training_history(history: list[dict], save_path: str | Path | None = None) -> None:
    epochs = [r["epoch"] for r in history]
    train_loss = [r["train_loss"] for r in history]
    val_f1 = [r["val_f1"] for r in history]
    val_acc = [r["val_acc"] for r in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, train_loss, label="Train Loss", color="tab:blue")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(epochs, val_f1, label="Val F1", color="tab:orange")
    axes[1].plot(epochs, val_acc, label="Val Acc", color="tab:green", linestyle="--")
    axes[1].set_title("Validation Metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Training curve → {save_path}")
    plt.show()


def plot_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str] | None = None,
    title: str = "Confusion Matrix",
    save_path: str | Path | None = None,
) -> None:
    if class_names is None:
        class_names = ["Non-stress", "Stress"]

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Confusion matrix → {save_path}")
    plt.show()


def plot_roc_curve(
    y_true: list[int],
    y_prob: list[float],
    label: str = "Model",
    save_path: str | Path | None = None,
) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# Error analysis
# ──────────────────────────────────────────────────────────────────────────────

def error_analysis(
    texts: list[str],
    y_true: list[int],
    y_pred: list[int],
    top_k: int = 20,
) -> dict[str, pd.DataFrame]:
    df = pd.DataFrame({"text": texts, "true": y_true, "pred": y_pred})
    fps = df[(df["true"] == 0) & (df["pred"] == 1)].head(top_k)
    fns = df[(df["true"] == 1) & (df["pred"] == 0)].head(top_k)
    return {"false_positives": fps, "false_negatives": fns}


# ──────────────────────────────────────────────────────────────────────────────
# Embedding visualization (t-SNE)
# ──────────────────────────────────────────────────────────────────────────────

def plot_tsne_embeddings(
    embeddings: np.ndarray,
    labels: list[int],
    domain_ids: list[int] | None = None,
    title: str = "Document Embeddings (t-SNE)",
    save_path: str | Path | None = None,
) -> None:
    from sklearn.manifold import TSNE

    print("  Running t-SNE …")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    emb_2d = tsne.fit_transform(embeddings)

    fig, axes = plt.subplots(1, 2 if domain_ids else 1, figsize=(14 if domain_ids else 7, 5))
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    # By label
    ax = axes[0]
    colors = ["tab:blue" if l == 0 else "tab:red" for l in labels]
    ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=colors, alpha=0.5, s=10)
    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(color="tab:blue", label="Non-stress"),
            Patch(color="tab:red", label="Stress"),
        ]
    )
    ax.set_title(f"{title} – by label")

    # By domain
    if domain_ids and len(axes) > 1:
        ax2 = axes[1]
        dcolors = ["tab:green" if d == 0 else "tab:orange" for d in domain_ids]
        ax2.scatter(emb_2d[:, 0], emb_2d[:, 1], c=dcolors, alpha=0.5, s=10)
        ax2.legend(
            handles=[
                Patch(color="tab:green", label="Dreaddit"),
                Patch(color="tab:orange", label="Counseling"),
            ]
        )
        ax2.set_title(f"{title} – by domain")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  t-SNE plot → {save_path}")
    plt.show()
