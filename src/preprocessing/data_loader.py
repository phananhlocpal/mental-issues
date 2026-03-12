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


def load_dreaddit(config: dict, split: str = "train") -> DomainDataset:
    """Load Dreaddit dataset from local CSV."""
    raw_dir = _resolve_path(config["data"]["raw_dir"])
    csv_path = raw_dir / f"dreaddit_{split}.csv"
    print(f"[Dreaddit] Loading split='{split}' from {csv_path} ...")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
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
    # Expect a 'text' column; fall back to first column
    text_col = "text" if "text" in df.columns else df.columns[0]
    label_col = "label" if "label" in df.columns else None
    texts  = [str(t) for t in df[text_col].tolist()]
    labels = [int(l) for l in df[label_col].tolist()] if label_col else [1] * len(texts)
    if max_samples:
        texts  = texts[:max_samples]
        labels = labels[:max_samples]
    return _preprocess_domain(texts, labels, domain_id=1, config=config)


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
        result["counseling"] = load_processed(cache_counseling)
    else:
        result["counseling"] = load_counseling(config, max_samples=2000)
        save_processed(result["counseling"], cache_counseling)

    return result


if __name__ == "__main__":
    cfg = load_config("config.yaml")
    datasets = prepare_all_datasets(cfg)
    for name, ds in datasets.items():
        print(f"  {name}: {len(ds)} samples | labels={set(ds.labels)}")
