"""Phase C3: feature-level sweeps for max_length and graph construction knobs.

Outputs:
- experiments/results/phase_c_max_length_impact.csv
- experiments/results/phase_c_graph_param_impact.csv
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graph import HeteroGraphBuilder
from src.preprocessing import build_dreaddit_protocol_splits, load_dreaddit
from src.utils import load_config, set_seed


def _truncate_texts(texts: list[str], max_tokens: int) -> list[str]:
    out = []
    for t in texts:
        toks = t.split()
        out.append(" ".join(toks[:max_tokens]))
    return out


def _eval_max_length(datasets: dict, splits: dict, max_len: int, seed: int) -> dict:
    tr = datasets["dreaddit_train"]
    te = datasets["dreaddit_test"]

    x_train = _truncate_texts([tr.clean_texts[i] for i in splits["train_idx"]], max_len)
    y_train = [tr.labels[i] for i in splits["train_idx"]]
    x_test = _truncate_texts(te.clean_texts, max_len)
    y_test = te.labels

    vec = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=2)
    xtr = vec.fit_transform(x_train)
    xte = vec.transform(x_test)

    clf = LogisticRegression(max_iter=1200, class_weight="balanced", random_state=seed)
    clf.fit(xtr, y_train)
    pred = clf.predict(xte)

    return {
        "max_length": int(max_len),
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1_macro": float(f1_score(y_test, pred, average="macro")),
        "f1_binary": float(f1_score(y_test, pred, average="binary", pos_label=1)),
        "recall_class1": float(recall_score(y_test, pred, pos_label=1)),
    }


def _eval_graph_params(cfg: dict, datasets: dict, tfidf_top_k: int, co_window: int) -> dict:
    c = copy.deepcopy(cfg)
    c.setdefault("preprocessing", {})["tfidf_top_k"] = int(tfidf_top_k)
    c.setdefault("preprocessing", {})["co_occurrence_window"] = int(co_window)

    t0 = time.time()
    g = HeteroGraphBuilder(c).build([datasets["dreaddit_train"], datasets["dreaddit_test"]])
    elapsed = time.time() - t0

    return {
        "tfidf_top_k": int(tfidf_top_k),
        "co_occurrence_window": int(co_window),
        "num_docs": int(g.num_docs),
        "num_words": int(g.num_words),
        "doc_word_edges": int(len(g.doc_word_edges)),
        "word_word_edges": int(len(g.word_word_edges)),
        "word_concept_edges": int(len(g.word_concept_edges)),
        "build_time_s": float(round(elapsed, 2)),
    }


def main() -> None:
    cfg = load_config(PROJECT_ROOT / "config.yaml")
    seed = int(cfg.get("training", {}).get("seed", 42))
    set_seed(seed)

    datasets = {
        "dreaddit_train": load_dreaddit(cfg, split="train"),
        "dreaddit_test": load_dreaddit(cfg, split="test"),
    }
    splits = build_dreaddit_protocol_splits(datasets, val_ratio=0.1, seed=seed)

    max_len_grid = [64, 128, 256, 512]
    max_len_rows = []
    for m in max_len_grid:
        print(f"[Phase C3] max_length={m}")
        max_len_rows.append(_eval_max_length(datasets, splits, m, seed))

    max_df = pd.DataFrame(max_len_rows)
    base = max_df.loc[max_df["max_length"] == 512].iloc[0]
    max_df["delta_f1_binary_vs_512"] = max_df["f1_binary"] - float(base["f1_binary"])
    max_df["delta_recall1_vs_512"] = max_df["recall_class1"] - float(base["recall_class1"])

    graph_grid = [
        (15, 3),
        (30, 5),
        (50, 8),
        (30, 8),
        (50, 5),
    ]
    graph_rows = []
    for tfidf_top_k, co_window in graph_grid:
        print(f"[Phase C3] graph tfidf_top_k={tfidf_top_k}, co_window={co_window}")
        graph_rows.append(_eval_graph_params(cfg, datasets, tfidf_top_k, co_window))

    graph_df = pd.DataFrame(graph_rows)

    out_dir = PROJECT_ROOT / "experiments" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    max_csv = out_dir / "phase_c_max_length_impact.csv"
    graph_csv = out_dir / "phase_c_graph_param_impact.csv"
    summary_json = out_dir / "phase_c_feature_sweep_summary.json"

    max_df.to_csv(max_csv, index=False)
    graph_df.to_csv(graph_csv, index=False)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "max_length_results": max_df.to_dict(orient="records"),
                "graph_param_results": graph_df.to_dict(orient="records"),
            },
            f,
            indent=2,
        )

    print(f"Saved: {max_csv}")
    print(f"Saved: {graph_csv}")
    print(f"Saved: {summary_json}")


if __name__ == "__main__":
    main()
