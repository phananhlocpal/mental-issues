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
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
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
    """Compute a comprehensive set of classification metrics.

    Returns accuracy, binary-F1, macro-F1, per-class P/R/F1,
    AUROC and AUPRC (when y_prob is supplied).
    """
    from sklearn.metrics import accuracy_score

    metrics: dict[str, float] = {
        "accuracy":           float(accuracy_score(y_true, y_pred)),
        "precision_binary":   float(precision_score(y_true, y_pred, average="binary", zero_division=0)),
        "recall_binary":      float(recall_score(y_true, y_pred, average="binary", zero_division=0)),
        "f1_binary":          float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
        "f1_macro":           float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro":    float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro":       float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    # per-class F1
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    for i, v in enumerate(per_class_f1):
        metrics[f"f1_class{i}"] = float(v)

    if y_prob is not None:
        try:
            metrics["auroc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            metrics["auroc"] = float("nan")
        try:
            metrics["auprc"] = float(average_precision_score(y_true, y_prob))
        except Exception:
            metrics["auprc"] = float("nan")

    # Legacy aliases so existing code that reads "precision"/"recall"/"f1" still works
    metrics["precision"] = metrics["precision_binary"]
    metrics["recall"]    = metrics["recall_binary"]
    metrics["f1"]        = metrics["f1_binary"]
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Bootstrap confidence intervals
# ──────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(
    y_true: list[int],
    y_pred: list[int],
    y_prob: list[float] | None = None,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, tuple[float, float]]:
    """Compute bootstrap confidence intervals for all metrics.

    Returns dict mapping metric_name → (lower_bound, upper_bound).
    """
    rng = np.random.default_rng(seed)
    y_true_a = np.array(y_true)
    y_pred_a = np.array(y_pred)
    y_prob_a = np.array(y_prob) if y_prob is not None else None

    boot_metrics: dict[str, list[float]] = {}
    n = len(y_true_a)

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y_true_a[idx].tolist()
        yp = y_pred_a[idx].tolist()
        ypr = y_prob_a[idx].tolist() if y_prob_a is not None else None
        m = compute_metrics(yt, yp, ypr)
        for k, v in m.items():
            boot_metrics.setdefault(k, []).append(v)

    alpha = (1.0 - ci) / 2.0
    result: dict[str, tuple[float, float]] = {}
    for k, vals in boot_metrics.items():
        arr = np.array(vals)
        result[k] = (float(np.nanquantile(arr, alpha)), float(np.nanquantile(arr, 1 - alpha)))
    return result


# ──────────────────────────────────────────────────────────────────────────────
# Multi-seed aggregation
# ──────────────────────────────────────────────────────────────────────────────

def aggregate_seed_results(
    seed_results: list[dict[str, float]],
) -> dict[str, dict[str, float]]:
    """Given a list of per-seed metric dicts, return mean ± std for each metric."""
    if not seed_results:
        return {}
    keys = seed_results[0].keys()
    agg: dict[str, dict[str, float]] = {}
    for k in keys:
        vals = np.array([r[k] for r in seed_results if not np.isnan(r.get(k, float("nan")))])
        if len(vals) == 0:
            agg[k] = {"mean": float("nan"), "std": float("nan"), "n": 0}
        else:
            agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
    return agg


def format_mean_std_table(
    agg_results: dict[str, dict[str, dict[str, float]]],
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Build a publication-ready mean±std table.

    Parameters
    ----------
    agg_results : {experiment_name: {metric: {mean, std, n}}}
    metrics     : subset of metric names to include; None = all
    """
    if metrics is None:
        all_keys: set[str] = set()
        for v in agg_results.values():
            all_keys.update(v.keys())
        metrics = sorted(all_keys)

    rows = []
    for exp_name, metric_agg in agg_results.items():
        row: dict[str, str | float] = {"experiment": exp_name}
        for m in metrics:
            if m in metric_agg:
                mean = metric_agg[m]["mean"]
                std  = metric_agg[m]["std"]
                row[m] = f"{mean:.4f}±{std:.4f}"
            else:
                row[m] = "N/A"
        rows.append(row)

    return pd.DataFrame(rows).set_index("experiment")


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


def plot_pr_curve(
    y_true: list[int],
    y_prob: list[float],
    label: str = "Model",
    save_path: str | Path | None = None,
) -> None:
    """Plot Precision-Recall curve with AUPRC score."""
    from sklearn.metrics import precision_recall_curve

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    plt.step(recall, precision, where="post", label=f"{label} (AP={auprc:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  PR curve → {save_path}")
    plt.show()


def plot_calibration_curve(
    y_true: list[int],
    y_prob: list[float],
    n_bins: int = 10,
    label: str = "Model",
    save_path: str | Path | None = None,
) -> None:
    """Reliability diagram (calibration curve)."""
    from sklearn.calibration import calibration_curve

    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )
    plt.figure(figsize=(6, 5))
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=label)
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration Curve")
    plt.legend()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Calibration curve → {save_path}")
    plt.show()


def plot_multi_roc(
    curves: list[tuple[list[int], list[float], str]],
    save_path: str | Path | None = None,
) -> None:
    """Overlay ROC curves for multiple experiments on one figure.

    curves: list of (y_true, y_prob, label)
    """
    plt.figure(figsize=(7, 6))
    for y_true, y_prob, label in curves:
        try:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc = roc_auc_score(y_true, y_prob)
            plt.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")
        except Exception:
            pass
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Comparison")
    plt.legend(fontsize=8)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Multi-ROC plot → {save_path}")
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# Error analysis
# ──────────────────────────────────────────────────────────────────────────────

def error_analysis(
    texts: list[str],
    y_true: list[int],
    y_pred: list[int],
    y_prob: list[float] | None = None,
    top_k: int = 20,
) -> dict[str, pd.DataFrame]:
    """Detailed error analysis with FP/FN taxonomy.

    Returns a dict with keys:
      - false_positives: predicted stress, actually non-stress
      - false_negatives: predicted non-stress, actually stress
      - correct_positive, correct_negative: correctly classified samples
      - summary: per-bucket counts and mean confidence (if y_prob supplied)
    """
    df = pd.DataFrame({"text": texts, "true": y_true, "pred": y_pred})
    if y_prob is not None:
        df["prob_stress"] = y_prob

    fps = df[(df["true"] == 0) & (df["pred"] == 1)].copy()
    fns = df[(df["true"] == 1) & (df["pred"] == 0)].copy()
    tps = df[(df["true"] == 1) & (df["pred"] == 1)].copy()
    tns = df[(df["true"] == 0) & (df["pred"] == 0)].copy()

    # Word-level taxonomy: count the most common tokens in FP vs FN buckets
    def top_words(texts_series: pd.Series, n: int = 15) -> list[str]:
        from collections import Counter
        import re
        tokens: list[str] = []
        for t in texts_series:
            tokens.extend(re.findall(r"\b\w{3,}\b", str(t).lower()))
        return [w for w, _ in Counter(tokens).most_common(n)]

    summary_rows = [
        {"bucket": "TP", "count": len(tps),
         "mean_prob": float(tps["prob_stress"].mean()) if y_prob is not None and len(tps) else float("nan"),
         "top_words": ", ".join(top_words(tps["text"])) if len(tps) else ""},
        {"bucket": "TN", "count": len(tns),
         "mean_prob": float(tns["prob_stress"].mean()) if y_prob is not None and len(tns) else float("nan"),
         "top_words": ", ".join(top_words(tns["text"])) if len(tns) else ""},
        {"bucket": "FP", "count": len(fps),
         "mean_prob": float(fps["prob_stress"].mean()) if y_prob is not None and len(fps) else float("nan"),
         "top_words": ", ".join(top_words(fps["text"])) if len(fps) else ""},
        {"bucket": "FN", "count": len(fns),
         "mean_prob": float(fns["prob_stress"].mean()) if y_prob is not None and len(fns) else float("nan"),
         "top_words": ", ".join(top_words(fns["text"])) if len(fns) else ""},
    ]

    return {
        "false_positives":  fps.head(top_k),
        "false_negatives":  fns.head(top_k),
        "correct_positive": tps.head(top_k),
        "correct_negative": tns.head(top_k),
        "summary":          pd.DataFrame(summary_rows),
    }


def save_error_analysis(
    analysis: dict[str, pd.DataFrame],
    out_dir: str | Path,
    prefix: str = "error_analysis",
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for key, df in analysis.items():
        path = out_dir / f"{prefix}_{key}.csv"
        df.to_csv(path, index=False)
    print(f"  Error analysis → {out_dir}/{prefix}_*.csv")


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
