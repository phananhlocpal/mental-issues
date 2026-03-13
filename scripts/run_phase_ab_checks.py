"""Run final Phase A/B checks and materialize reproducibility artifacts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_eda_report import _pick_text_col, summarize_csv
from src.preprocessing import (
    build_dreaddit_protocol_splits,
    prepare_all_datasets,
    save_protocol_splits,
)
from src.utils import clean_text
from src.utils import load_config, set_seed


def _build_eda_report(project_root: Path) -> dict:
    raw = project_root / "data" / "raw"
    datasets = {
        "dreaddit_train": raw / "dreaddit_train.csv",
        "dreaddit_test": raw / "dreaddit_test.csv",
        "counseling": raw / "counseling.csv",
    }
    report = {name: summarize_csv(path, name) for name, path in datasets.items()}

    tr = pd.read_csv(datasets["dreaddit_train"])
    te = pd.read_csv(datasets["dreaddit_test"])
    tr_set = set(tr[_pick_text_col(tr)].fillna("").astype(str).map(clean_text))
    te_set = set(te[_pick_text_col(te)].fillna("").astype(str).map(clean_text))
    overlap = len(tr_set.intersection(te_set))
    report["dreaddit_overlap"] = {
        "clean_text_intersection": int(overlap),
        "dreaddit_test_unique": int(len(te_set)),
        "overlap_ratio_vs_test_unique": float(overlap / max(1, len(te_set))),
    }
    return report


def _write_eda_outputs(report: dict, project_root: Path) -> dict[str, str]:
    out_dir = project_root / "experiments" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / "eda_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    rows = []
    for name, stats in report.items():
        if isinstance(stats, dict) and "rows" in stats:
            rows.append(
                {
                    "dataset": name,
                    "rows": stats["rows"],
                    "text_col": stats["text_col"],
                    "label_col": stats["label_col"],
                    "empty_after_clean": stats["empty_after_clean"],
                    "duplicate_clean_text": stats["duplicate_clean_text"],
                    "len_words_mean": round(float(stats["len_words_mean"]), 3),
                    "len_words_median": round(float(stats["len_words_median"]), 3),
                    "len_words_p95": round(float(stats["len_words_p95"]), 3),
                }
            )
    summary_path = out_dir / "eda_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    return {
        "report_json": str(report_path),
        "summary_csv": str(summary_path),
    }


def main() -> None:
    config = load_config(str(PROJECT_ROOT / "config.yaml"))
    seed = int(config.get("training", {}).get("seed", 42))
    val_size = float(config.get("training", {}).get("val_size", 0.1))
    set_seed(seed)

    # Phase A: EDA report + quality gates at load time.
    datasets = prepare_all_datasets(config, force=False)
    report = _build_eda_report(PROJECT_ROOT)
    outputs = _write_eda_outputs(report, PROJECT_ROOT)

    # Phase B: official in-domain split artifact.
    splits = build_dreaddit_protocol_splits(
        datasets,
        val_ratio=val_size,
        seed=seed,
    )
    split_path = PROJECT_ROOT / "experiments" / "results" / "dreaddit_protocol_splits.json"
    save_protocol_splits(splits, split_path)

    summary = {
        "eda_outputs": outputs,
        "split_output": str(split_path),
        "split_sizes": {k: int(v.shape[0]) for k, v in splits.items()},
        "counseling_labeled": bool(datasets["counseling"].is_labeled),
    }
    summary_path = PROJECT_ROOT / "experiments" / "results" / "phase_ab_status.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Phase A/B status -> {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
