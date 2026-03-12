"""Main experiment runner.

Usage:
    python run_experiments.py --exp all          # run all experiments
    python run_experiments.py --exp in_domain
    python run_experiments.py --exp cross_domain
    python run_experiments.py --exp ablation
    python run_experiments.py --exp baseline
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import load_config, set_seed, get_device
from src.preprocessing import prepare_all_datasets, DomainDataset
from src.entity_extraction import MedicalEntityExtractor
from src.graph import HeteroGraphBuilder, save_graph, load_graph, build_node_features
from src.models import MentalHealthGNN, BERTBaseline
from src.training import GNNTrainer
from src.evaluation import (
    compute_metrics,
    build_performance_table,
    print_performance_table,
    save_performance_table,
    plot_training_history,
    plot_confusion_matrix,
    plot_tsne_embeddings,
    error_analysis,
)
from src.explainability import MentalHealthExplainer


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def split_indices(n: int, val_ratio: float = 0.1, test_ratio: float = 0.2, seed: int = 42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)
    return idx[n_test + n_val:], idx[n_val:n_test + n_val], idx[:n_test]


def make_model(config: dict, input_dims: dict[str, int], use_da: bool, metadata):
    mcfg = config["model"]
    return MentalHealthGNN(
        input_dims=input_dims,
        hidden_dim=mcfg["hidden_dim"],
        num_heads=mcfg["num_heads"],
        num_layers=mcfg.get("num_layers", 2),
        dropout=mcfg["dropout"],
        num_classes=mcfg["num_classes"],
        num_domains=2,
        use_domain_adversarial=use_da,
        metadata=metadata,
    )


def get_pyg_metadata(pyg_data):
    """Extract (node_types, edge_types) metadata tuple from PyG HeteroData."""
    return pyg_data.metadata()


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1: Build graph & node features
# ──────────────────────────────────────────────────────────────────────────────

def stage_build_graph(config: dict, datasets: dict, force: bool = False):
    graph_dir = Path(config["data"]["graph_dir"])
    graph_path = graph_dir / "hetero_graph.pkl"
    features_path = graph_dir / "node_features.pt"

    if not force and graph_path.exists():
        print("[Stage] Loading cached graph …")
        graph = load_graph(graph_path)
    else:
        print("[Stage] Building heterogeneous graph …")
        builder = HeteroGraphBuilder(config)
        ds_list = [datasets["dreaddit_train"], datasets["counseling"]]
        graph = builder.build(ds_list)
        save_graph(graph, graph_path)

    if not force and features_path.exists():
        print("[Stage] Loading cached node features …")
        node_feats = torch.load(features_path, map_location="cpu")
    else:
        print("[Stage] Computing node features …")
        ds_list = [datasets["dreaddit_train"], datasets["counseling"]]
        node_feats = build_node_features(graph, ds_list, config, device="cpu")
        torch.save(node_feats, features_path)
        print(f"  Node features → {features_path}")

    return graph, node_feats


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2: Convert to PyG + attach features
# ──────────────────────────────────────────────────────────────────────────────

def stage_to_pyg(graph, node_feats: dict[str, torch.Tensor], config: dict, device: torch.device):
    from src.graph.graph_builder import HeteroGraphBuilder
    builder = HeteroGraphBuilder(config)
    pyg_data = builder.to_pyg(graph, device=device)

    proj_dim = config["embeddings"]["projection_dim"]

    # Attach feature tensors
    for ntype, feat in node_feats.items():
        if hasattr(pyg_data[ntype], "num_nodes"):
            pyg_data[ntype].x = feat.to(device)

    # Add reverse edges for message passing
    from torch_geometric.transforms import ToUndirected
    # Manual reverse edges
    if ("document", "contains", "word") in pyg_data.edge_types:
        ei = pyg_data["document", "contains", "word"].edge_index
        pyg_data["word", "rev_contains", "document"].edge_index = ei.flip(0)

    if ("word", "maps_to", "medical_concept") in pyg_data.edge_types:
        ei = pyg_data["word", "maps_to", "medical_concept"].edge_index
        pyg_data["medical_concept", "rev_maps_to", "word"].edge_index = ei.flip(0)

    if ("medical_concept", "belongs_to", "symptom_category") in pyg_data.edge_types:
        ei = pyg_data["medical_concept", "belongs_to", "symptom_category"].edge_index
        pyg_data["symptom_category", "rev_belongs_to", "medical_concept"].edge_index = ei.flip(0)

    input_dims = {ntype: node_feats[ntype].shape[1] for ntype in node_feats}
    return pyg_data, input_dims


# ──────────────────────────────────────────────────────────────────────────────
# Experiment 1: In-domain
# ──────────────────────────────────────────────────────────────────────────────

def run_in_domain(config, pyg_data, graph, device, input_dims) -> dict:
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: In-Domain (Dreaddit → Dreaddit)")
    print("=" * 60)
    set_seed(config["training"]["seed"])

    labels = pyg_data["document"].y.cpu()
    domain_ids = pyg_data["document"].domain.cpu()

    # Use only dreaddit docs (domain_id == 0)
    dreaddit_mask = (domain_ids == 0).nonzero(as_tuple=True)[0]
    n = len(dreaddit_mask)
    train_rel, val_rel, test_rel = split_indices(n, seed=config["training"]["seed"])
    train_idx = dreaddit_mask[train_rel]
    val_idx = dreaddit_mask[val_rel]
    test_idx = dreaddit_mask[test_rel]

    model = make_model(config, input_dims, use_da=False, metadata=pyg_data.metadata())
    trainer = GNNTrainer(model, config, device, run_name="exp1_in_domain")

    history = trainer.fit(
        pyg_data,
        train_indices=train_idx,
        val_indices=val_idx,
        labels=labels,
        domain_labels=domain_ids,
    )
    trainer.load_best()
    plot_training_history(
        history,
        save_path=f"{config['experiments']['result_dir']}/exp1_training_curve.png",
    )

    # Test evaluation
    test_stats = trainer.evaluate(
        pyg_data, test_idx, labels, domain_ids, config["training"]["batch_size"]
    )
    print(f"  Test F1={test_stats['f1']:.4f} | Acc={test_stats['accuracy']:.4f}")
    return test_stats


# ──────────────────────────────────────────────────────────────────────────────
# Experiment 2: Cross-domain
# ──────────────────────────────────────────────────────────────────────────────

def run_cross_domain(config, pyg_data, graph, device, input_dims) -> dict:
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Cross-Domain (Dreaddit ↔ Counseling)")
    print("=" * 60)
    set_seed(config["training"]["seed"])

    labels = pyg_data["document"].y.cpu()
    domain_ids = pyg_data["document"].domain.cpu()

    dreaddit_mask = (domain_ids == 0).nonzero(as_tuple=True)[0]
    counseling_mask = (domain_ids == 1).nonzero(as_tuple=True)[0]

    results = {}

    for train_domain, test_mask, exp_name in [
        (dreaddit_mask, counseling_mask, "exp2a_dreaddit2counseling"),
        (counseling_mask, dreaddit_mask, "exp2b_counseling2dreaddit"),
    ]:
        print(f"\n  Run: {exp_name}")
        n = len(train_domain)
        train_rel, val_rel, _ = split_indices(n, seed=config["training"]["seed"])
        train_idx = train_domain[train_rel]
        val_idx = train_domain[val_rel]

        model = make_model(config, input_dims, use_da=True, metadata=pyg_data.metadata())
        trainer = GNNTrainer(model, config, device, run_name=exp_name)

        trainer.fit(pyg_data, train_idx, val_idx, labels, domain_ids)
        trainer.load_best()

        test_stats = trainer.evaluate(
            pyg_data, test_mask, labels, domain_ids, config["training"]["batch_size"]
        )
        print(f"  Test F1={test_stats['f1']:.4f} | Acc={test_stats['accuracy']:.4f}")
        results[exp_name] = test_stats

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Experiment 3: Ablation
# ──────────────────────────────────────────────────────────────────────────────

def run_ablation(config, pyg_data, graph, device, input_dims) -> dict:
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Ablation Study")
    print("=" * 60)

    labels = pyg_data["document"].y.cpu()
    domain_ids = pyg_data["document"].domain.cpu()
    dreaddit_mask = (domain_ids == 0).nonzero(as_tuple=True)[0]
    n = len(dreaddit_mask)
    train_rel, val_rel, test_rel = split_indices(n, seed=config["training"]["seed"])

    ablation_results: dict[str, dict] = {}
    variants = config["experiments"].get("ablation_variants", ["full_model"])

    for variant in variants:
        print(f"\n  Variant: {variant}")
        set_seed(config["training"]["seed"])

        # Modify config per variant
        use_da = "no_domain_adversarial" not in variant
        model = make_model(config, input_dims, use_da=use_da, metadata=pyg_data.metadata())
        run_name = f"ablation_{variant}"
        trainer = GNNTrainer(model, config, device, run_name=run_name)

        trainer.fit(
            pyg_data,
            dreaddit_mask[train_rel],
            dreaddit_mask[val_rel],
            labels,
            domain_ids,
        )
        trainer.load_best()
        test_stats = trainer.evaluate(
            pyg_data,
            dreaddit_mask[test_rel],
            labels,
            domain_ids,
            config["training"]["batch_size"],
        )
        ablation_results[variant] = test_stats
        print(f"  F1={test_stats['f1']:.4f}")

    return ablation_results


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp",
        choices=["all", "in_domain", "cross_domain", "ablation", "build_graph"],
        default="all",
    )
    parser.add_argument("--force", action="store_true", help="Force re-build cache")
    args = parser.parse_args()

    config = load_config("config.yaml")
    set_seed(config["training"]["seed"])
    device = get_device(config["training"]["device"])
    print(f"Device: {device}")

    # ── Load data ───────────────────────────────────────────────────────
    print("\n[Step 1] Loading datasets …")
    datasets = prepare_all_datasets(config, force=args.force)
    for name, ds in datasets.items():
        print(f"  {name}: {len(ds)} samples")

    if args.exp == "build_graph":
        stage_build_graph(config, datasets, force=True)
        return

    # ── Build graph ─────────────────────────────────────────────────────
    print("\n[Step 2] Building heterogeneous graph …")
    graph, node_feats = stage_build_graph(config, datasets, force=args.force)
    print(graph.summary())

    # ── Convert to PyG ──────────────────────────────────────────────────
    print("\n[Step 3] Converting to PyG …")
    pyg_data, input_dims = stage_to_pyg(graph, node_feats, config, device)
    print(f"  Node types: {pyg_data.node_types}")
    print(f"  Edge types: {pyg_data.edge_types}")

    # ── Run experiments ──────────────────────────────────────────────────
    all_results: dict[str, dict] = {}

    if args.exp in ("all", "in_domain"):
        r = run_in_domain(config, pyg_data, graph, device, input_dims)
        all_results["in_domain"] = r

    if args.exp in ("all", "cross_domain"):
        r = run_cross_domain(config, pyg_data, graph, device, input_dims)
        all_results.update(r)

    if args.exp in ("all", "ablation"):
        r = run_ablation(config, pyg_data, graph, device, input_dims)
        all_results.update(r)

    # ── Summary table ────────────────────────────────────────────────────
    if all_results:
        df = build_performance_table(all_results)
        print_performance_table(df)
        result_dir = Path(config["experiments"]["result_dir"])
        result_dir.mkdir(parents=True, exist_ok=True)
        save_performance_table(df, result_dir / "performance_table.csv")

        # Save JSON
        with open(result_dir / "all_results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  Results JSON → {result_dir / 'all_results.json'}")


if __name__ == "__main__":
    main()
