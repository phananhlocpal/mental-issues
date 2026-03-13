"""Phase E smoke test – validates metrics, CI, aggregation, error_analysis."""
import numpy as np
import torch

from src.evaluation.metrics import (
    compute_metrics,
    bootstrap_ci,
    aggregate_seed_results,
    format_mean_std_table,
    error_analysis,
    save_error_analysis,
)

# ----- compute_metrics -----
y_true = [0, 0, 1, 1, 1, 0, 1, 0, 1, 1]
y_pred = [0, 1, 1, 0, 1, 0, 1, 1, 1, 0]
y_prob = [0.1, 0.8, 0.9, 0.3, 0.85, 0.2, 0.7, 0.6, 0.95, 0.4]

m = compute_metrics(y_true, y_pred, y_prob)
assert "f1_binary"   in m, "missing f1_binary"
assert "f1_macro"    in m, "missing f1_macro"
assert "f1_class0"   in m, "missing f1_class0"
assert "f1_class1"   in m, "missing f1_class1"
assert "auroc"       in m, "missing auroc"
assert "auprc"       in m, "missing auprc"
assert "precision"   in m, "missing legacy precision alias"
assert "f1"          in m, "missing legacy f1 alias"
print(f"compute_metrics OK: f1_binary={m['f1_binary']:.4f} auroc={m['auroc']:.4f} auprc={m['auprc']:.4f}")

# ----- bootstrap_ci -----
ci = bootstrap_ci(y_true, y_pred, y_prob, n_bootstrap=200, seed=42)
assert "f1_binary" in ci
lo, hi = ci["f1_binary"]
assert lo <= hi, f"CI inverted: {lo} > {hi}"
print(f"bootstrap_ci OK: f1_binary 95% CI = ({lo:.4f}, {hi:.4f})")

# ----- aggregate_seed_results -----
seed_results = [
    compute_metrics(y_true, y_pred, y_prob),
    compute_metrics(y_true, y_pred, y_prob),
    compute_metrics(y_true, y_pred, y_prob),
]
agg = aggregate_seed_results(seed_results)
assert "f1_binary" in agg
assert agg["f1_binary"]["n"] == 3
assert agg["f1_binary"]["std"] == 0.0   # identical runs → std=0
print(f"aggregate_seed_results OK: f1_binary mean={agg['f1_binary']['mean']:.4f} std={agg['f1_binary']['std']:.4f}")

# ----- format_mean_std_table -----
table = format_mean_std_table(
    {"variant_a": agg, "variant_b": agg},
    metrics=["f1_binary", "auroc", "auprc"],
)
assert "variant_a" in table.index
assert "variant_b" in table.index
print("format_mean_std_table OK")
print(table)

# ----- error_analysis -----
texts = [f"sample text {i}" for i in range(10)]
ea = error_analysis(texts, y_true, y_pred, y_prob, top_k=5)
assert "false_positives" in ea
assert "false_negatives" in ea
assert "summary"         in ea
print(f"error_analysis OK: {len(ea['false_positives'])} FPs, {len(ea['false_negatives'])} FNs")
print(ea["summary"])

# ----- save_error_analysis -----
import tempfile, pathlib
with tempfile.TemporaryDirectory() as td:
    save_error_analysis(ea, td, prefix="test")
    files = list(pathlib.Path(td).glob("test_*.csv"))
    assert len(files) >= 3, f"Expected >=3 CSVs, got {len(files)}"
    print(f"save_error_analysis OK: {len(files)} files written")

print("\nPhase E smoke test PASSED")
