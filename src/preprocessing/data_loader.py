"""Data loading and preprocessing for mental health detection."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

# Project root = two levels up from this file (src/preprocessing/data_loader.py)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_path(p: str | Path) -> Path:
    """Resolve a path: if absolute, use as-is; if relative, resolve from project root."""
    p = Path(p)
    return p if p.is_absolute() else _PROJECT_ROOT / p

import nltk
import numpy as np
import pandas as pd
from datasets import Dataset
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm
from transformers import AutoTokenizer

from src.utils import clean_text, load_config

# Download NLTK data (run once)
def _download_nltk() -> None:
    # punkt_tab is required by NLTK >= 3.9; punkt is the legacy name
    for resource, category in [
        ("stopwords",  "corpora"),
        ("wordnet",    "corpora"),
        ("omw-1.4",   "corpora"),
        ("punkt",      "tokenizers"),
        ("punkt_tab",  "tokenizers"),
    ]:
        try:
            nltk.data.find(f"{category}/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


_download_nltk()


# ──────────────────────────────────────────────────────────────────────────────
# Dataset loaders
# ──────────────────────────────────────────────────────────────────────────────

class DomainDataset:
    """Unified wrapper around one domain's split data."""

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        domain_ids: list[int],
        clean_texts: list[str],
        tokens: list[list[str]],
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.domain_ids = domain_ids
        self.clean_texts = clean_texts
        self.tokens = tokens

    def __len__(self) -> int:
        return len(self.texts)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "text": self.texts,
                "label": self.labels,
                "domain_id": self.domain_ids,
                "clean_text": self.clean_texts,
            }
        )


def _quality_cfg(config: dict) -> dict:
    """Return data-quality settings with safe defaults."""
    dq = config.get("data_quality", {})
    return {
        "enabled": dq.get("enabled", True),
        "max_duplicate_ratio": float(dq.get("max_duplicate_ratio", 0.80)),
        "max_empty_ratio": float(dq.get("max_empty_ratio", 0.05)),
        "deduplicate_clean_text": bool(dq.get("deduplicate_clean_text", False)),
        "allow_single_class_for": set(dq.get("allow_single_class_for", ["counseling"])),
        "require_binary_labels_for": set(
            dq.get("require_binary_labels_for", ["dreaddit_train", "dreaddit_test"])
        ),
    }


def _validate_and_filter_dataframe(
    df: pd.DataFrame,
    dataset_name: str,
    text_col: str,
    label_col: str,
    config: dict,
) -> pd.DataFrame:
    """Apply data gates and optional de-duplication to a loaded dataframe."""
    dq = _quality_cfg(config)
    if not dq["enabled"]:
        return df

    work = df.copy()
    work[text_col] = work[text_col].fillna("").astype(str)
    work[label_col] = work[label_col].astype(int)
    work["_clean_text"] = work[text_col].map(clean_text)

    n = max(1, len(work))
    empty_ratio = float((work["_clean_text"].str.len() == 0).sum()) / n
    dup_ratio = float(work["_clean_text"].duplicated().sum()) / n

    if empty_ratio > dq["max_empty_ratio"]:
        raise ValueError(
            f"[{dataset_name}] empty text ratio {empty_ratio:.2%} exceeds "
            f"max_empty_ratio={dq['max_empty_ratio']:.2%}"
        )
    if dup_ratio > dq["max_duplicate_ratio"]:
        raise ValueError(
            f"[{dataset_name}] duplicate clean-text ratio {dup_ratio:.2%} exceeds "
            f"max_duplicate_ratio={dq['max_duplicate_ratio']:.2%}"
        )

    labels = set(work[label_col].dropna().astype(int).tolist())
    if not labels:
        raise ValueError(f"[{dataset_name}] empty label set after loading")

    if dataset_name in dq["require_binary_labels_for"] and not labels.issubset({0, 1}):
        raise ValueError(f"[{dataset_name}] expects binary labels {{0,1}} but found {sorted(labels)}")

    if len(labels) < 2 and dataset_name not in dq["allow_single_class_for"]:
        raise ValueError(
            f"[{dataset_name}] single-class dataset is not allowed by data_quality policy: "
            f"labels={sorted(labels)}"
        )

    if dq["deduplicate_clean_text"]:
        before = len(work)
        work = work.drop_duplicates(subset=["_clean_text"], keep="first")
        after = len(work)
        if before != after:
            print(f"[{dataset_name}] deduplicated by clean text: {before} -> {after}")

    return work.drop(columns=["_clean_text"])


def load_dreaddit(config: dict, split: str = "train") -> DomainDataset:
    """Load Dreaddit dataset from local CSV."""
    raw_dir = _resolve_path(config["data"]["raw_dir"])
    csv_path = raw_dir / f"dreaddit_{split}.csv"
    print(f"[Dreaddit] Loading split='{split}' from {csv_path} ...")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    df = _validate_and_filter_dataframe(
        df=df,
        dataset_name=f"dreaddit_{split}",
        text_col="text",
        label_col="label",
        config=config,
    )
    texts  = [str(t) for t in df["text"].tolist()]
    labels = [int(l) for l in df["label"].tolist()]
    return _preprocess_domain(texts, labels, domain_id=0, config=config)


def load_counseling(config: dict, max_samples: Optional[int] = None) -> DomainDataset:
    """Load Mental Health Counseling dataset from local CSV."""
    raw_dir = _resolve_path(config["data"]["raw_dir"])
    csv_path = raw_dir / "counseling.csv"
    print(f"[Counseling] Loading from {csv_path} ...")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    text_col = "text" if "text" in df.columns else ("context" if "context" in df.columns else None)
    label_col = "label" if "label" in df.columns else None
    if text_col is None:
        raise ValueError("[counseling] missing required text column ('text' or 'context')")
    if label_col is None:
        raise ValueError("[counseling] missing required 'label' column")

    df = _validate_and_filter_dataframe(
        df=df,
        dataset_name="counseling",
        text_col=text_col,
        label_col=label_col,
        config=config,
    )
    texts  = [str(t) for t in df[text_col].tolist()]
    labels = [int(l) for l in df[label_col].tolist()]
    if max_samples:
        texts  = texts[:max_samples]
        labels = labels[:max_samples]
    return _preprocess_domain(texts, labels, domain_id=1, config=config)


def stratified_train_val_split_indices(
    labels: list[int] | np.ndarray,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a deterministic stratified train/val split from label array."""
    y = np.asarray(labels)
    if y.ndim != 1:
        raise ValueError("labels must be 1-D")
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be in (0, 1)")

    rng = np.random.default_rng(seed)
    train_chunks: list[np.ndarray] = []
    val_chunks: list[np.ndarray] = []

    for c in np.unique(y):
        cls_idx = np.where(y == c)[0]
        rng.shuffle(cls_idx)
        n_val = max(1, int(round(len(cls_idx) * val_ratio)))
        if n_val >= len(cls_idx):
            n_val = max(1, len(cls_idx) - 1)
        val_chunks.append(cls_idx[:n_val])
        train_chunks.append(cls_idx[n_val:])

    train_idx = np.concatenate(train_chunks) if train_chunks else np.array([], dtype=int)
    val_idx = np.concatenate(val_chunks) if val_chunks else np.array([], dtype=int)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx.astype(int), val_idx.astype(int)


def build_dreaddit_protocol_splits(
    datasets: dict[str, DomainDataset],
    val_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Protocol split: train/val from dreaddit_train, test from dreaddit_test."""
    if "dreaddit_train" not in datasets or "dreaddit_test" not in datasets:
        raise KeyError("datasets must include both 'dreaddit_train' and 'dreaddit_test'")

    train_ds = datasets["dreaddit_train"]
    test_ds = datasets["dreaddit_test"]
    tr_idx, va_idx = stratified_train_val_split_indices(
        labels=train_ds.labels,
        val_ratio=val_ratio,
        seed=seed,
    )
    te_idx = np.arange(len(test_ds), dtype=int)
    return {
        "train_idx": tr_idx,
        "val_idx": va_idx,
        "test_idx": te_idx,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────────────────────────────────────

def _preprocess_domain(
    texts: list[str],
    labels: list[int],
    domain_id: int,
    config: dict,
) -> DomainDataset:
    """Clean → tokenize → lemmatize a list of raw texts."""
    lemmatizer = WordNetLemmatizer()
    sw = set(stopwords.words("english"))

    clean_texts: list[str] = []
    token_lists: list[list[str]] = []

    for text in tqdm(texts, desc=f"  Preprocessing domain={domain_id}", leave=False):
        ct = clean_text(text)
        try:
            tokens = nltk.word_tokenize(ct)
        except LookupError:
            # Fallback: simple whitespace split if punkt data unavailable
            tokens = ct.split()
        # Stopword removal + lemmatization
        tokens = [
            lemmatizer.lemmatize(t)
            for t in tokens
            if t not in sw and len(t) > 1
        ]
        clean_texts.append(ct)
        token_lists.append(tokens)

    domain_ids = [domain_id] * len(texts)
    return DomainDataset(texts, labels, domain_ids, clean_texts, token_lists)


# ──────────────────────────────────────────────────────────────────────────────
# BERT Tokenization helper
# ──────────────────────────────────────────────────────────────────────────────

class BERTTokenizedDataset:
    """Wraps a DomainDataset and provides BERT tokenization on demand."""

    def __init__(
        self,
        domain_dataset: DomainDataset,
        model_name: str = "bert-base-uncased",
        max_length: int = 512,
    ) -> None:
        self.ds = domain_dataset
        self.max_length = max_length
        print(f"  Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_batch(self, texts: list[str]) -> dict:
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def get_all_encodings(self) -> dict:
        return self.tokenize_batch(self.ds.clean_texts)


# ──────────────────────────────────────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────────────────────────────────────

def save_processed(dataset: DomainDataset, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(dataset, f)
    print(f"  Saved → {path}")


def load_processed(path: str | Path) -> DomainDataset:
    with open(path, "rb") as f:
        return pickle.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# Main helper
# ──────────────────────────────────────────────────────────────────────────────

def prepare_all_datasets(config: dict, force: bool = False) -> dict[str, DomainDataset]:
    """Load and preprocess all configured datasets, with disk caching."""
    proc_dir = _resolve_path(config["data"]["processed_dir"])
    raw_dir = _resolve_path(config["data"]["raw_dir"])
    result: dict[str, DomainDataset] = {}

    cache_dreaddit_train = proc_dir / "dreaddit_train.pkl"
    cache_dreaddit_test = proc_dir / "dreaddit_test.pkl"
    cache_counseling = proc_dir / "counseling.pkl"

    if not force and cache_dreaddit_train.exists():
        print("[Cache] Loading Dreaddit train from disk …")
        result["dreaddit_train"] = load_processed(cache_dreaddit_train)
    else:
        result["dreaddit_train"] = load_dreaddit(config, split="train")
        save_processed(result["dreaddit_train"], cache_dreaddit_train)

    if not force and cache_dreaddit_test.exists():
        print("[Cache] Loading Dreaddit test from disk …")
        result["dreaddit_test"] = load_processed(cache_dreaddit_test)
    else:
        result["dreaddit_test"] = load_dreaddit(config, split="test")
        save_processed(result["dreaddit_test"], cache_dreaddit_test)

    if not force and cache_counseling.exists():
        print("[Cache] Loading Counseling from disk …")
        cached = load_processed(cache_counseling)

        # Backward-compat: older pipeline cached only first 2000 samples.
        # If cache is truncated, rebuild full counseling dataset automatically.
        counseling_csv = raw_dir / "counseling.csv"
        refresh_needed = False
        if counseling_csv.exists():
            try:
                n_raw = len(pd.read_csv(counseling_csv))
                if len(cached) < n_raw:
                    refresh_needed = True
                    print(
                        f"[Cache] Counseling cache appears truncated "
                        f"({len(cached)}/{n_raw}); rebuilding full dataset …"
                    )
            except Exception:
                # If row count check fails, keep cache for robustness.
                pass

        if refresh_needed:
            result["counseling"] = load_counseling(config, max_samples=None)
            save_processed(result["counseling"], cache_counseling)
        else:
            result["counseling"] = cached
    else:
        result["counseling"] = load_counseling(config, max_samples=None)
        save_processed(result["counseling"], cache_counseling)

    return result


if __name__ == "__main__":
    cfg = load_config("config.yaml")
    datasets = prepare_all_datasets(cfg)
    for name, ds in datasets.items():
        print(f"  {name}: {len(ds)} samples | labels={set(ds.labels)}")
