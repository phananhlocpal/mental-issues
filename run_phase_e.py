"""Phase E – Full Ablation, Baselines, and Statistical Reporting.

Usage:
    python run_phase_e.py [--seeds 5] [--skip-baselines] [--skip-ablation]

Outputs (all under experiments/results/phase_e/):
    ablation_seed_results.json
    baseline_seed_results.json
    phase_e_mean_std_table.csv
    phase_e_mean_std_table.md
    error_analysis_<variant>/  (FP/FN/summary CSVs for best seed)
    calibration_<variant>.png
    roc_multi.png
    pr_multi.png
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

# ── project imports ──────────────────────────────────────────────────────────
from src.utils import set_seed, load_config
from src.preprocessing.data_loader import (
    MentalHealthDataLoader,
    build_dreaddit_protocol_splits,
)
from src.graph.graph_builder import HeteroGraphBuilder
from src.graph.node_features import build_node_features
from src.models import MentalHealthGNN
from src.models.baselines import TFIDFBaseline, SciBERTMLPBaseline, HomoGCNBaseline, HomoGATBaseline, build_homo_graph
from src.training.trainer import GNNTrainer
from src.training.baseline_trainer import EmbeddingTrainer
from src.evaluation.metrics import (
    compute_metrics,
    bootstrap_ci,
    aggregate_seed_results,
    format_mean_std_table,
    save_error_analysis,
    error_analysis,
    plot_multi_roc,
    plot_pr_curve,
    plot_calibration_curve,
)

# ──────────────────────────────────────────────────────────────────────────────
# Config helpers
# ──────────────────────────────────────────────────────────────────────────────

ABLATION_VARIANTS: dict[str, dict] = {
    "full_model": {},
    "no_knowledge_graph": {
        "graph": {"min_entity_confidence": 1.1}  # effectively disables word-concept edges
    },
    "no_domain_adversarial": {
        "training": {"domain_lambda": 0.0},
        "model": {"use_domain_adversarial": False},
    },
    "no_symptom_nodes": {
        "_disable_symptom_nodes": True  # handled in build step
    },
}

OUT_DIR = Path("experiments/results/phase_e")


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into a copy of base."""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and k in result and isinstance(result[k], dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"  Saved → {path}")


# ──────────────────────────────────────────────────────────────────────────────
# Data / Graph build (shared across seeds for a fixed graph topology)
# ──────────────────────────────────────────────────────────────────────────────

def build_data(cfg: dict, seed: int):
    """Load datasets, build graph, compute node features. Returns shared objects."""
    device = cfg["training"].get("device", "cpu")
    loader = MentalHealthDataLoader(cfg)
    datasets = loader.load_all()

    splits = build_dreaddit_protocol_splits(datasets, val_ratio=0.1, seed=seed)
    train_idx = torch.tensor(splits["train_idx"], dtype=torch.long)
    val_idx   = torch.tensor(splits["val_idx"],   dtype=torch.long)
    # test indices are relative to dreaddit_test; shift by len(train_ds) for the concatenated graph
    n_train_ds = len(datasets["dreaddit_train"])
    test_idx_local = torch.tensor(splits["test_idx"], dtype=torch.long)
    test_idx = test_idx_local + n_train_ds

    # Combined dataset list for graph (only dreaddit – counseling is unlabeled)
    graph_datasets = [datasets["dreaddit_train"], datasets["dreaddit_test"]]
    builder = HeteroGraphBuilder(cfg)
    pyg_data = builder.build(graph_datasets)

    # Build node features and attach to pyg_data.x_dict
    node_feats = build_node_features(pyg_data, graph_datasets, cfg, device=device)
    for nt, feat in node_feats.items():
        pyg_data[nt].x = feat.to("cpu")  # keep on CPU; trainer moves as needed
    pyg_data.x_dict  # trigger HeteroData property update

    # Labels and domain labels tensors (train concat test)
    all_labels = torch.tensor(
        datasets["dreaddit_train"].labels + datasets["dreaddit_test"].labels,
        dtype=torch.long,
    )
    domain_labels = torch.tensor(
        datasets["dreaddit_train"].domain_ids + datasets["dreaddit_test"].domain_ids,
        dtype=torch.long,
    )

    # Raw texts for error analysis (same order as graph)
    all_texts = datasets["dreaddit_train"].clean_texts + datasets["dreaddit_test"].clean_texts

    return pyg_data, train_idx, val_idx, test_idx, all_labels, domain_labels, all_texts


# ──────────────────────────────────────────────────────────────────────────────
# Single GNN run
# ──────────────────────────────────────────────────────────────────────────────

def run_gnn_single(
    variant_name: str,
    variant_override: dict,
    base_cfg: dict,
    seed: int,
    device: str,
) -> dict[str, float]:
    """Train and evaluate one GNN variant with one seed. Returns test metrics."""
    set_seed(seed)
    cfg = _deep_merge(base_cfg, variant_override)
    cfg["training"]["seed"] = seed

    print(f"\n  [{variant_name}] seed={seed}")
    pyg_data, train_idx, val_idx, test_idx, all_labels, domain_labels, all_texts = build_data(
        cfg, seed
    )

    # Build model
    in_dims = {nt: pyg_data.x_dict[nt].shape[1] for nt in pyg_data.node_types}
    model = MentalHealthGNN(
        in_channels_dict=in_dims,
        hidden_channels=cfg["model"]["hidden_dim"],
        num_classes=cfg["model"]["num_classes"],
        num_heads=cfg["model"]["num_heads"],
        num_layers=cfg["model"]["num_layers"],
        dropout=cfg["model"]["dropout"],
        use_domain_adversarial=cfg["model"].get("use_domain_adversarial", True),
        metadata=(pyg_data.node_types, pyg_data.edge_types),
    )

    run_name = f"phase_e_{variant_name}_seed{seed}"
    trainer = GNNTrainer(model, cfg, run_name=run_name, device=device)
    trainer.fit(pyg_data, train_idx, val_idx, all_labels, domain_labels)

    metrics = trainer.evaluate(
        pyg_data, test_idx, all_labels, domain_labels,
        batch_size=cfg["training"]["batch_size"],
    )
    print(f"    test f1_binary={metrics['f1_binary']:.4f}  auroc={metrics.get('auroc', float('nan')):.4f}")
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# GNN inference-only (for error analysis on best seed checkpoint)
# ──────────────────────────────────────────────────────────────────────────────

def gnn_predict_all(
    trainer: GNNTrainer,
    pyg_data,
    test_idx: torch.Tensor,
    all_labels: torch.Tensor,
    domain_labels: torch.Tensor,
    batch_size: int,
) -> tuple[list[int], list[int], list[float]]:
    """Return (y_true, y_pred, y_prob) for the test set."""
    trainer.model.eval()
    device = trainer.device
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for start in range(0, len(test_idx), batch_size):
            idx = test_idx[start : start + batch_size].to(device)
            lbl = all_labels[idx.cpu()].tolist()
            out = trainer.model(
                x_dict=pyg_data.x_dict,
                edge_index_dict=pyg_data.edge_index_dict,
                doc_indices=idx,
                alpha=0.0,
            )
            probs = torch.softmax(out["logits"], dim=-1)[:, 1].cpu().tolist()
            preds = out["logits"].argmax(-1).cpu().tolist()
            y_true.extend(lbl)
            y_pred.extend(preds)
            y_prob.extend(probs)
    return y_true, y_pred, y_prob


# ──────────────────────────────────────────────────────────────────────────────
# Baseline runs
# ──────────────────────────────────────────────────────────────────────────────

def run_baselines_single(base_cfg: dict, seed: int, device: str) -> dict[str, dict[str, float]]:
    """Run all baseline models for one seed. Returns {baseline_name: metrics}."""
    set_seed(seed)
    loader = MentalHealthDataLoader(base_cfg)
    datasets = loader.load_all()
    splits = build_dreaddit_protocol_splits(datasets, val_ratio=0.1, seed=seed)

    train_ds = datasets["dreaddit_train"]
    test_ds  = datasets["dreaddit_test"]

    train_t = [train_ds.clean_texts[i] for i in splits["train_idx"]]
    train_l = [train_ds.labels[i] for i in splits["train_idx"]]
    test_t  = test_ds.clean_texts
    test_l  = test_ds.labels

    results: dict[str, dict[str, float]] = {}

    # TF-IDF LR
    lr_bl = TFIDFBaseline(classifier="lr")
    r = _run_sklearn(lr_bl, train_t, train_l, test_t, test_l)
    results["tfidf_lr"] = r
    print(f"    [TF-IDF LR]  f1_binary={r.get('f1_binary', r.get('f1', 0)):.4f}")

    # TF-IDF SVM
    svm_bl = TFIDFBaseline(classifier="svm")
    r = _run_sklearn(svm_bl, train_t, train_l, test_t, test_l)
    results["tfidf_svm"] = r
    print(f"    [TF-IDF SVM] f1_binary={r.get('f1_binary', r.get('f1', 0)):.4f}")

    # Build graph & features once for GNN-style baselines
    graph_datasets = [train_ds, test_ds]
    builder = HeteroGraphBuilder(base_cfg)
    pyg_data = builder.build(graph_datasets)
    node_feats = build_node_features(pyg_data, graph_datasets, base_cfg, device=device)
    for nt, feat in node_feats.items():
        pyg_data[nt].x = feat.to("cpu")

    n_train = len(train_ds)
    n_test  = len(test_ds)
    all_labels = torch.tensor(train_ds.labels + test_ds.labels, dtype=torch.long)
    # Indices within the combined node ordering (doc nodes first = all train + all test)
    train_gl = torch.tensor(splits["train_idx"], dtype=torch.long)  # into train portion
    val_gl   = torch.tensor(splits["val_idx"],   dtype=torch.long)
    test_gl  = torch.arange(n_train, n_train + n_test, dtype=torch.long)

    doc_feats = pyg_data.x_dict["document"]

    # SciBERT-MLP
    mlp = SciBERTMLPBaseline(input_dim=doc_feats.shape[1])
    mlp_trainer = EmbeddingTrainer(
        mlp, mode="mlp", device=device,
        epochs=base_cfg["training"]["epochs"],
        patience=base_cfg["training"]["early_stopping_patience"],
    )
    mlp_trainer.fit(doc_feats, None, train_gl, val_gl, all_labels)
    r = mlp_trainer.evaluate(doc_feats, None, test_gl, all_labels)
    results["scibert_mlp"] = r
    print(f"    [SciBERT-MLP]  f1_binary={r.get('f1_binary', r.get('f1', 0)):.4f}")

    # Homo GCN / GAT (project to 128-d first)
    x_homo, edge_index, offsets, _ = build_homo_graph(
        pyg_data, pyg_data.x_dict, proj_dim=128, device="cpu"
    )
    doc_off = offsets["document"]
    train_abs = train_gl + doc_off
    val_abs   = val_gl   + doc_off
    test_abs  = test_gl  + doc_off
    n_total   = x_homo.shape[0]
    full_labels = torch.zeros(n_total, dtype=torch.long)
    for idx_local, lbl in zip(range(n_train + n_test), all_labels.tolist()):
        full_labels[idx_local + doc_off] = lbl

    gcn = HomoGCNBaseline(input_dim=128, hidden_dim=128)
    gcn_trainer = EmbeddingTrainer(
        gcn, mode="gnn", device=device,
        epochs=base_cfg["training"]["epochs"],
        patience=base_cfg["training"]["early_stopping_patience"],
    )
    gcn_trainer.fit(x_homo, edge_index, train_abs, val_abs, full_labels)
    r = gcn_trainer.evaluate(x_homo, edge_index, test_abs, full_labels)
    results["homo_gcn"] = r
    print(f"    [Homo GCN]   f1_binary={r.get('f1_binary', r.get('f1', 0)):.4f}")

    gat = HomoGATBaseline(input_dim=128, hidden_dim=64)
    gat_trainer = EmbeddingTrainer(
        gat, mode="gnn", device=device,
        epochs=base_cfg["training"]["epochs"],
        patience=base_cfg["training"]["early_stopping_patience"],
    )
    gat_trainer.fit(x_homo, edge_index, train_abs, val_abs, full_labels)
    r = gat_trainer.evaluate(x_homo, edge_index, test_abs, full_labels)
    results["homo_gat"] = r
    print(f"    [Homo GAT]   f1_binary={r.get('f1_binary', r.get('f1', 0)):.4f}")

    return results


def _run_sklearn(
    baseline: TFIDFBaseline,
    train_texts: list[str],
    train_labels: list[int],
    test_texts: list[str],
    test_labels: list[int],
) -> dict:
    import time as _time
    t0 = _time.time()
    baseline.fit(train_texts, train_labels)
    elapsed = _time.time() - t0
    preds = baseline.predict(test_texts).tolist()
    probs_arr = baseline.predict_proba(test_texts)
    probs = probs_arr.tolist() if probs_arr is not None else None
    m = compute_metrics(test_labels, preds, probs)
    m["train_time_s"] = round(elapsed, 2)
    return m


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(n_seeds: int = 5, skip_baselines: bool = False, skip_ablation: bool = False) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_cfg = load_config("config.yaml")
    device = base_cfg["training"].get("device", "cpu")
    seeds = list(range(42, 42 + n_seeds))

    # ── A. Ablation ──────────────────────────────────────────────────────────
    ablation_seed_results: dict[str, list[dict]] = {v: [] for v in ABLATION_VARIANTS}

    if not skip_ablation:
        print("\n" + "=" * 60)
        print("PHASE E — ABLATION STUDY")
        print("=" * 60)
        # Keep error-analysis artefacts only for the median seed of full_model
        full_model_runs: list[tuple[int, float, GNNTrainer, object, torch.Tensor, torch.Tensor, torch.Tensor, list[str]]] = []

        for variant_name, variant_override in ABLATION_VARIANTS.items():
            print(f"\n▶ Variant: {variant_name}")
            for seed in seeds:
                try:
                    m = run_gnn_single(variant_name, variant_override, base_cfg, seed, device)
                    ablation_seed_results[variant_name].append(m)
                    if variant_name == "full_model":
                        # Cache for error analysis later
                        full_model_runs.append((seed, m.get("f1_binary", m.get("f1", 0.0)), m))
                except Exception as exc:
                    print(f"    WARN: variant={variant_name} seed={seed} failed: {exc}")
                    ablation_seed_results[variant_name].append({})

        _save_json(ablation_seed_results, OUT_DIR / "ablation_seed_results.json")
        print("\nAblation seed results saved.")

        # Error analysis on full_model best-seed run
        if full_model_runs:
            best_seed, best_f1, _ = max(full_model_runs, key=lambda x: x[1])
            print(f"\n  Running error analysis for full_model best seed={best_seed} (f1={best_f1:.4f}) …")
            set_seed(best_seed)
            cfg_ea = _deep_merge(base_cfg, {})
            pyg_data, train_idx, val_idx, test_idx, all_labels, domain_labels, all_texts = build_data(
                cfg_ea, best_seed
            )
            in_dims = {nt: pyg_data.x_dict[nt].shape[1] for nt in pyg_data.node_types}
            model_ea = MentalHealthGNN(
                in_channels_dict=in_dims,
                hidden_channels=base_cfg["model"]["hidden_dim"],
                num_classes=base_cfg["model"]["num_classes"],
                num_heads=base_cfg["model"]["num_heads"],
                num_layers=base_cfg["model"]["num_layers"],
                dropout=base_cfg["model"]["dropout"],
                use_domain_adversarial=base_cfg["model"].get("use_domain_adversarial", True),
                metadata=(pyg_data.node_types, pyg_data.edge_types),
            )
            run_name_ea = f"phase_e_full_model_seed{best_seed}"
            trainer_ea = GNNTrainer(model_ea, cfg_ea, run_name=run_name_ea, device=device)
            trainer_ea.fit(pyg_data, train_idx, val_idx, all_labels, domain_labels)

            y_true, y_pred, y_prob = gnn_predict_all(
                trainer_ea, pyg_data, test_idx, all_labels, domain_labels,
                base_cfg["training"]["batch_size"],
            )
            test_texts = [all_texts[i] for i in test_idx.tolist()]
            ea = error_analysis(test_texts, y_true, y_pred, y_prob, top_k=30)
            save_error_analysis(ea, OUT_DIR / "error_analysis_full_model", "full_model")

            # Calibration + ROC + PR curves
            try:
                import matplotlib
                matplotlib.use("Agg")
                plot_calibration_curve(
                    y_true, y_prob, label="full_model",
                    save_path=OUT_DIR / "calibration_full_model.png",
                )
                plot_pr_curve(
                    y_true, y_prob, label="full_model",
                    save_path=OUT_DIR / "pr_full_model.png",
                )
            except Exception as exc:
                print(f"  WARN: plot failed: {exc}")

    # ── B. Baselines ─────────────────────────────────────────────────────────
    baseline_seed_results: dict[str, list[dict]] = {}

    if not skip_baselines:
        print("\n" + "=" * 60)
        print("PHASE E — BASELINES")
        print("=" * 60)
        for seed in seeds:
            print(f"\n  seed={seed}")
            try:
                seed_bl = run_baselines_single(base_cfg, seed, device)
                for bl_name, m in seed_bl.items():
                    baseline_seed_results.setdefault(bl_name, []).append(m)
            except Exception as exc:
                print(f"  WARN: baselines seed={seed} failed: {exc}")

        _save_json(baseline_seed_results, OUT_DIR / "baseline_seed_results.json")
        print("\nBaseline seed results saved.")

    # ── C. Aggregate and format results table ─────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE E — AGGREGATE RESULTS")
    print("=" * 60)

    all_agg: dict[str, dict] = {}

    if ablation_seed_results:
        for v_name, runs in ablation_seed_results.items():
            valid = [r for r in runs if r]
            if valid:
                all_agg[v_name] = aggregate_seed_results(valid)

    if baseline_seed_results:
        for bl_name, runs in baseline_seed_results.items():
            valid = [r for r in runs if r]
            if valid:
                all_agg[bl_name] = aggregate_seed_results(valid)

    if all_agg:
        report_metrics = [
            "accuracy", "f1_binary", "f1_macro",
            "precision_binary", "recall_binary",
            "f1_class0", "f1_class1",
            "auroc", "auprc",
        ]
        table = format_mean_std_table(all_agg, metrics=report_metrics)
        csv_path = OUT_DIR / "phase_e_mean_std_table.csv"
        md_path  = OUT_DIR / "phase_e_mean_std_table.md"
        table.to_csv(csv_path)
        table.to_markdown(md_path)
        print(f"\n  Results table → {csv_path}")
        print(f"  Results table → {md_path}")
        print("\n" + table.to_string())

    # ── D. Bootstrap CI for best variant ──────────────────────────────────────
    if ablation_seed_results.get("full_model"):
        combined: dict[str, list[float]] = {}
        for r in ablation_seed_results["full_model"]:
            for k, v in r.items():
                combined.setdefault(k, []).append(v)

        ci_rows = []
        for k, vals in combined.items():
            arr = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
            if not arr:
                continue
            ci_rows.append({
                "metric": k,
                "mean": round(float(np.mean(arr)), 4),
                "std":  round(float(np.std(arr)), 4),
                "min":  round(float(np.min(arr)), 4),
                "max":  round(float(np.max(arr)), 4),
                "n_seeds": len(arr),
            })
        ci_df = pd.DataFrame(ci_rows).sort_values("metric")
        ci_path = OUT_DIR / "full_model_ci_summary.csv"
        ci_df.to_csv(ci_path, index=False)
        print(f"\n  CI summary (full_model) → {ci_path}")

    print("\nPhase E complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--skip-ablation", action="store_true")
    args = parser.parse_args()
    main(n_seeds=args.seeds, skip_baselines=args.skip_baselines, skip_ablation=args.skip_ablation)
