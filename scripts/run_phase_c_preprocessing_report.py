"""Phase C: compare preprocessing variants with a fast in-domain baseline."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import build_dreaddit_protocol_splits, load_dreaddit
from src.utils import load_config, set_seed


def _eval_variant(cfg: dict, name: str, seed: int) -> dict:
    set_seed(seed)
    datasets = {
        "dreaddit_train": load_dreaddit(cfg, split="train"),
        "dreaddit_test": load_dreaddit(cfg, split="test"),
    }
    splits = build_dreaddit_protocol_splits(datasets, val_ratio=0.1, seed=seed)

    train_ds = datasets["dreaddit_train"]
    test_ds = datasets["dreaddit_test"]

    x_train = [train_ds.clean_texts[i] for i in splits["train_idx"]]
    y_train = [train_ds.labels[i] for i in splits["train_idx"]]
    x_test = test_ds.clean_texts
    y_test = test_ds.labels

    vec = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=2)
    xtr = vec.fit_transform(x_train)
    xte = vec.transform(x_test)

    clf = LogisticRegression(max_iter=1200, class_weight="balanced", random_state=seed)
    clf.fit(xtr, y_train)
    pred = clf.predict(xte)

    return {
        "variant": name,
        "dreaddit_train_total": int(len(train_ds)),
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1_macro": float(f1_score(y_test, pred, average="macro")),
        "f1_binary": float(f1_score(y_test, pred, average="binary", pos_label=1)),
        "recall_class1": float(recall_score(y_test, pred, pos_label=1)),
    }


def _variant(base_cfg: dict, name: str, **pp_overrides) -> tuple[str, dict]:
    cfg = copy.deepcopy(base_cfg)
    pp = cfg.setdefault("preprocessing", {})
    for k, v in pp_overrides.items():
        pp[k] = v
    return name, cfg


def main() -> None:
    cfg = load_config(PROJECT_ROOT / "config.yaml")
    seed = int(cfg.get("training", {}).get("seed", 42))

    variants: list[tuple[str, dict]] = [
        ("default", copy.deepcopy(cfg)),
        _variant(cfg, "no_stopword_removal", remove_stopwords=False),
        _variant(cfg, "no_lemmatization", use_lemmatization=False),
        _variant(cfg, "no_repeat_char_collapse", collapse_repeated_chars=False),
        _variant(cfg, "near_dedup_090", train_dedup_near=True, train_dedup_near_threshold=0.90),
        _variant(cfg, "near_dedup_093", train_dedup_near=True, train_dedup_near_threshold=0.93),
        _variant(cfg, "near_dedup_095", train_dedup_near=True, train_dedup_near_threshold=0.95),
    ]

    rows = []
    for name, vcfg in variants:
        print(f"[Phase C] evaluating variant: {name}")
        rows.append(_eval_variant(vcfg, name, seed))

    df = pd.DataFrame(rows)
    base = df.loc[df["variant"] == "default"].iloc[0]
    df["delta_f1_binary_vs_default"] = df["f1_binary"] - float(base["f1_binary"])
    df["delta_recall1_vs_default"] = df["recall_class1"] - float(base["recall_class1"])
    df = df.sort_values(by=["f1_binary", "recall_class1"], ascending=False).reset_index(drop=True)

    out_dir = PROJECT_ROOT / "experiments" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "phase_c_preprocessing_impact.csv"
    json_path = out_dir / "phase_c_preprocessing_impact.json"
    top_path = out_dir / "phase_c_preprocessing_top3.csv"

    df.to_csv(csv_path, index=False)
    df.head(3).to_csv(top_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2)

    print(f"Phase C preprocessing report -> {csv_path}")
    print(f"Phase C top-3 variants -> {top_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
