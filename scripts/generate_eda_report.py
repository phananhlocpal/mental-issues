from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils import clean_text


def _pick_text_col(df: pd.DataFrame) -> str:
    lut = {c.lower(): c for c in df.columns}
    for c in ("text", "context"):
        if c in lut:
            return lut[c]
    if "response" in lut:
        return lut["response"]
    raise ValueError(f"Cannot infer text column from columns={list(df.columns)}")


def summarize_csv(path: Path, name: str) -> dict:
    df = pd.read_csv(path)
    text_col = _pick_text_col(df)
    label_col = "label" if "label" in df.columns else None

    text = df[text_col].fillna("").astype(str)
    clean = text.map(clean_text)
    lengths = clean.str.split().map(len)

    out = {
        "dataset": name,
        "path": str(path),
        "rows": int(len(df)),
        "columns": list(df.columns),
        "text_col": text_col,
        "label_col": label_col,
        "empty_after_clean": int((clean.str.len() == 0).sum()),
        "duplicate_raw_text": int(text.duplicated().sum()),
        "duplicate_clean_text": int(clean.duplicated().sum()),
        "len_words_mean": float(lengths.mean()),
        "len_words_median": float(lengths.median()),
        "len_words_p95": float(lengths.quantile(0.95)),
    }

    if label_col:
        out["label_dist"] = {str(k): int(v) for k, v in df[label_col].value_counts(dropna=False).to_dict().items()}
        out["label_unique"] = sorted([int(x) for x in df[label_col].dropna().unique().tolist()])
    else:
        out["label_dist"] = None
        out["label_unique"] = None

    return out


def main() -> None:
    root = ROOT
    raw = root / "data" / "raw"
    out_dir = root / "experiments" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "dreaddit_train": raw / "dreaddit_train.csv",
        "dreaddit_test": raw / "dreaddit_test.csv",
        "counseling": raw / "counseling.csv",
    }

    report = {name: summarize_csv(path, name) for name, path in datasets.items()}

    tr = pd.read_csv(datasets["dreaddit_train"])
    te = pd.read_csv(datasets["dreaddit_test"])
    tr_col = _pick_text_col(tr)
    te_col = _pick_text_col(te)
    tr_set = set(tr[tr_col].fillna("").astype(str).map(clean_text))
    te_set = set(te[te_col].fillna("").astype(str).map(clean_text))
    overlap = len(tr_set.intersection(te_set))
    report["dreaddit_overlap"] = {
        "clean_text_intersection": int(overlap),
        "dreaddit_test_unique": int(len(te_set)),
        "overlap_ratio_vs_test_unique": float(overlap / max(1, len(te_set))),
    }

    report_path = out_dir / "eda_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    rows = []
    for k, v in report.items():
        if isinstance(v, dict) and "rows" in v:
            rows.append(
                {
                    "dataset": k,
                    "rows": v["rows"],
                    "text_col": v["text_col"],
                    "label_col": v["label_col"],
                    "empty_after_clean": v["empty_after_clean"],
                    "duplicate_clean_text": v["duplicate_clean_text"],
                    "len_words_mean": round(v["len_words_mean"], 3),
                    "len_words_median": round(v["len_words_median"], 3),
                    "len_words_p95": round(v["len_words_p95"], 3),
                }
            )
    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(out_dir / "eda_summary.csv", index=False)

    print(f"EDA report saved: {report_path}")
    print(f"EDA summary saved: {out_dir / 'eda_summary.csv'}")


if __name__ == "__main__":
    main()
