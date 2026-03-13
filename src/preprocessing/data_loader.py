"""Data loading and preprocessing for mental health detection."""
from __future__ import annotations

import json
import pickle
import html
import re
import unicodedata
from difflib import SequenceMatcher
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

# ──────────────────────────────────────────────────────────────────────────────
# LIWC / DAL psycholinguistic auxiliary feature columns
# These features ship with Dreaddit and directly encode stress-relevant cues.
# ──────────────────────────────────────────────────────────────────────────────
LIWC_FEATURE_COLS: list[str] = [
    # LIWC summary dimensions
    "lex_liwc_Analytic", "lex_liwc_Clout", "lex_liwc_Authentic", "lex_liwc_Tone",
    # Affect
    "lex_liwc_affect", "lex_liwc_posemo", "lex_liwc_negemo",
    "lex_liwc_anx", "lex_liwc_anger", "lex_liwc_sad",
    # Social / interpersonal
    "lex_liwc_social", "lex_liwc_family", "lex_liwc_friend",
    # Personal pronouns (self-focus is a robust stress marker)
    "lex_liwc_i", "lex_liwc_we", "lex_liwc_you",
    # Cognitive processing
    "lex_liwc_cogproc", "lex_liwc_insight", "lex_liwc_cause", "lex_liwc_discrep",
    "lex_liwc_tentat", "lex_liwc_certain",
    # Biological / health
    "lex_liwc_bio", "lex_liwc_body", "lex_liwc_health",
    # Temporal focus
    "lex_liwc_focuspast", "lex_liwc_focuspresent", "lex_liwc_focusfuture",
    # DAL valence/arousal
    "lex_dal_avg_pleasantness", "lex_dal_avg_activation", "lex_dal_avg_imagery",
    # Sentiment + readability
    "sentiment", "syntax_ari",
]

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
        is_labeled: bool = True,
        aux_features: Optional[np.ndarray] = None,
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.domain_ids = domain_ids
        self.clean_texts = clean_texts
        self.tokens = tokens
        self.is_labeled = is_labeled
        # (N, n_aux) float32 array of LIWC/DAL/sentiment features, or None
        self.aux_features = aux_features

    def __setstate__(self, state):
        """Backward-compat for old pickles saved before new fields existed."""
        self.__dict__.update(state)
        if "is_labeled" not in self.__dict__:
            self.is_labeled = True
        if "aux_features" not in self.__dict__:
            self.aux_features = None

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
        "allow_unlabeled_for": set(dq.get("allow_unlabeled_for", ["counseling"])),
        "require_binary_labels_for": set(
            dq.get("require_binary_labels_for", ["dreaddit_train", "dreaddit_test"])
        ),
        "max_dreaddit_train_test_overlap_ratio": float(
            dq.get("max_dreaddit_train_test_overlap_ratio", 0.02)
        ),
    }


def _preprocessing_cfg(config: dict) -> dict:
    pp = config.get("preprocessing", {})
    return {
        "remove_stopwords": bool(pp.get("remove_stopwords", True)),
        "use_lemmatization": bool(pp.get("use_lemmatization", True)),
        "normalize_html": bool(pp.get("normalize_html", True)),
        "normalize_unicode": bool(pp.get("normalize_unicode", True)),
        "collapse_repeated_chars": bool(pp.get("collapse_repeated_chars", True)),
        "repeated_char_max_run": int(pp.get("repeated_char_max_run", 2)),
        "train_dedup_exact": bool(pp.get("train_dedup_exact", True)),
        "train_dedup_near": bool(pp.get("train_dedup_near", False)),
        "train_dedup_near_threshold": float(pp.get("train_dedup_near_threshold", 0.95)),
    }


_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _normalize_text_for_preprocessing(text: str, config: dict) -> str:
    """Normalize malformed text before canonical clean_text()."""
    pp = _preprocessing_cfg(config)
    s = str(text)

    if pp["normalize_unicode"]:
        s = unicodedata.normalize("NFKC", s)
    if pp["normalize_html"]:
        s = html.unescape(s)
        s = _HTML_TAG_RE.sub(" ", s)
    if pp["collapse_repeated_chars"]:
        max_run = max(1, pp["repeated_char_max_run"])
        # Reduce elongated tokens like "soooo" -> "soo" when max_run=2.
        s = re.sub(r"(.)\1{%d,}" % max_run, lambda m: m.group(1) * max_run, s)

    return clean_text(s)


def _deduplicate_train_examples(
    texts: list[str],
    labels: list[int],
    config: dict,
) -> tuple[list[str], list[int], dict]:
    """Drop exact/near duplicates from training data only."""
    pp = _preprocessing_cfg(config)
    if not texts:
        return texts, labels, {"dropped_exact": 0, "dropped_near": 0}

    clean_texts = [_normalize_text_for_preprocessing(t, config) for t in texts]

    keep_idx: list[int] = []
    seen_exact: set[str] = set()
    dropped_exact = 0
    if pp["train_dedup_exact"]:
        for i, ct in enumerate(clean_texts):
            if ct in seen_exact:
                dropped_exact += 1
                continue
            seen_exact.add(ct)
            keep_idx.append(i)
    else:
        keep_idx = list(range(len(texts)))

    dropped_near = 0
    if pp["train_dedup_near"] and keep_idx:
        threshold = pp["train_dedup_near_threshold"]
        buckets: dict[tuple[int, str], list[str]] = {}
        filtered_idx: list[int] = []

        for i in keep_idx:
            ct = clean_texts[i]
            key = (len(ct) // 20, ct[:24])
            candidates = buckets.get(key, [])
            is_dup = any(SequenceMatcher(None, ct, prev).ratio() >= threshold for prev in candidates)
            if is_dup:
                dropped_near += 1
                continue
            filtered_idx.append(i)
            buckets.setdefault(key, []).append(ct)

        keep_idx = filtered_idx

    dedup_texts = [texts[i] for i in keep_idx]
    dedup_labels = [labels[i] for i in keep_idx]
    stats = {
        "dropped_exact": int(dropped_exact),
        "dropped_near": int(dropped_near),
    }
    return dedup_texts, dedup_labels, stats, keep_idx


def _compute_clean_text_overlap_ratio(train_texts: list[str], test_texts: list[str]) -> float:
    """Return overlap ratio of clean texts: |train ∩ test| / |test_unique|."""
    train_set = set(train_texts)
    test_set = set(test_texts)
    if not test_set:
        return 0.0
    return float(len(train_set.intersection(test_set))) / float(len(test_set))


def _validate_and_filter_dataframe(
    df: pd.DataFrame,
    dataset_name: str,
    text_col: str,
    label_col: Optional[str],
    config: dict,
) -> pd.DataFrame:
    """Apply data gates and optional de-duplication to a loaded dataframe."""
    dq = _quality_cfg(config)
    if not dq["enabled"]:
        return df

    work = df.copy()
    work[text_col] = work[text_col].fillna("").astype(str)
    has_labels = label_col is not None and label_col in work.columns
    if has_labels:
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

    if has_labels:
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
    else:
        if dataset_name not in dq["allow_unlabeled_for"]:
            raise ValueError(
                f"[{dataset_name}] missing labels but dataset is not allowed in unlabeled mode"
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

    # ── Extract LIWC / DAL auxiliary features if enabled ──────────────
    use_liwc = config.get("preprocessing", {}).get("use_liwc_features", False)
    aux_mat: Optional[np.ndarray] = None
    if use_liwc:
        avail = [c for c in LIWC_FEATURE_COLS if c in df.columns]
        if avail:
            raw = df[avail].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            aux_mat = raw.values.astype(np.float32)
        else:
            print(f"[dreaddit_{split}] use_liwc_features=true but no LIWC columns found.")

    if split == "train":
        texts, labels, dedup_stats, keep_idx = _deduplicate_train_examples(texts, labels, config)
        if dedup_stats["dropped_exact"] > 0 or dedup_stats["dropped_near"] > 0:
            print(
                "[dreaddit_train] de-dup applied: "
                f"exact={dedup_stats['dropped_exact']}, near={dedup_stats['dropped_near']}"
            )
        if aux_mat is not None:
            aux_mat = aux_mat[np.array(keep_idx)]

    ds = _preprocess_domain(texts, labels, domain_id=0, config=config)
    ds.aux_features = aux_mat
    return ds


def load_counseling(config: dict, max_samples: Optional[int] = None) -> DomainDataset:
    """Load Mental Health Counseling dataset from local CSV."""
    raw_dir = _resolve_path(config["data"]["raw_dir"])
    csv_path = raw_dir / "counseling.csv"
    print(f"[Counseling] Loading from {csv_path} ...")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)

    col_lut = {c.lower(): c for c in df.columns}
    text_col = None
    for key in ("text", "context"):
        if key in col_lut:
            text_col = col_lut[key]
            break
    response_col = col_lut.get("response")
    if text_col is None and response_col is not None:
        text_col = response_col

    label_col = "label" if "label" in df.columns else None
    if text_col is None:
        raise ValueError("[counseling] missing required text column ('text' or 'context')")

    adaptation_only = label_col is None
    if adaptation_only:
        print("[Counseling] No label column found. Loading as adaptation-only (labels=-1).")

    df = _validate_and_filter_dataframe(
        df=df,
        dataset_name="counseling",
        text_col=text_col,
        label_col=label_col,
        config=config,
    )
    texts  = [str(t) for t in df[text_col].tolist()]
    labels = [int(l) for l in df[label_col].tolist()] if label_col else [-1] * len(texts)
    if max_samples:
        texts  = texts[:max_samples]
        labels = labels[:max_samples]
    ds = _preprocess_domain(texts, labels, domain_id=1, config=config)
    ds.is_labeled = not adaptation_only
    return ds


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


def save_protocol_splits(splits: dict[str, np.ndarray], path: str | Path) -> None:
    """Persist protocol split indices as JSON for reproducibility."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: [int(i) for i in v.tolist()] for k, v in splits.items()}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  Protocol splits → {path}")


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
    pp = _preprocessing_cfg(config)
    lemmatizer = WordNetLemmatizer()
    sw = set(stopwords.words("english"))

    clean_texts: list[str] = []
    token_lists: list[list[str]] = []

    for text in tqdm(texts, desc=f"  Preprocessing domain={domain_id}", leave=False):
        ct = _normalize_text_for_preprocessing(text, config)
        try:
            tokens = nltk.word_tokenize(ct)
        except LookupError:
            # Fallback: simple whitespace split if punkt data unavailable
            tokens = ct.split()
        tokens = [t for t in tokens if len(t) > 1]
        if pp["remove_stopwords"]:
            tokens = [t for t in tokens if t not in sw]
        if pp["use_lemmatization"]:
            tokens = [lemmatizer.lemmatize(t) for t in tokens]
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

    use_liwc = config.get("preprocessing", {}).get("use_liwc_features", False)

    def _liwc_stale(cached: DomainDataset) -> bool:
        """True when LIWC is enabled but cached dataset has no aux_features."""
        return use_liwc and getattr(cached, "aux_features", None) is None

    if not force and cache_dreaddit_train.exists():
        print("[Cache] Loading Dreaddit train from disk …")
        cached_train = load_processed(cache_dreaddit_train)
        if _liwc_stale(cached_train):
            print("[Cache] Dreaddit train cache missing LIWC features; rebuilding …")
            result["dreaddit_train"] = load_dreaddit(config, split="train")
            save_processed(result["dreaddit_train"], cache_dreaddit_train)
        else:
            result["dreaddit_train"] = cached_train
    else:
        result["dreaddit_train"] = load_dreaddit(config, split="train")
        save_processed(result["dreaddit_train"], cache_dreaddit_train)

    if not force and cache_dreaddit_test.exists():
        print("[Cache] Loading Dreaddit test from disk …")
        cached_test = load_processed(cache_dreaddit_test)
        if _liwc_stale(cached_test):
            print("[Cache] Dreaddit test cache missing LIWC features; rebuilding …")
            result["dreaddit_test"] = load_dreaddit(config, split="test")
            save_processed(result["dreaddit_test"], cache_dreaddit_test)
        else:
            result["dreaddit_test"] = cached_test
    else:
        result["dreaddit_test"] = load_dreaddit(config, split="test")
        save_processed(result["dreaddit_test"], cache_dreaddit_test)

    if not force and cache_counseling.exists():
        print("[Cache] Loading Counseling from disk …")
        cached = load_processed(cache_counseling)

        # Refresh when cache is stale or schema/label mode changed.
        counseling_csv = raw_dir / "counseling.csv"
        refresh_needed = False
        if counseling_csv.exists():
            try:
                csv_df = pd.read_csv(counseling_csv)
                lower_cols = {c.lower() for c in csv_df.columns}
                n_raw = len(pd.read_csv(counseling_csv))
                if len(cached) < n_raw:
                    refresh_needed = True
                    print(
                        f"[Cache] Counseling cache appears truncated "
                        f"({len(cached)}/{n_raw}); rebuilding full dataset …"
                    )

                csv_has_label = "label" in lower_cols
                cached_is_labeled = bool(getattr(cached, "is_labeled", True))
                if cached_is_labeled != csv_has_label:
                    refresh_needed = True
                    print(
                        "[Cache] Counseling label mode changed "
                        f"(cached_is_labeled={cached_is_labeled}, csv_has_label={csv_has_label}); "
                        "rebuilding cache …"
                    )

                if cache_counseling.stat().st_mtime < counseling_csv.stat().st_mtime:
                    refresh_needed = True
                    print("[Cache] Counseling CSV is newer than cache; rebuilding …")
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

    # Gate Dreaddit train-test leakage by cleaned text overlap.
    dq = _quality_cfg(config)
    if dq["enabled"] and "dreaddit_train" in result and "dreaddit_test" in result:
        overlap_ratio = _compute_clean_text_overlap_ratio(
            result["dreaddit_train"].clean_texts,
            result["dreaddit_test"].clean_texts,
        )
        if overlap_ratio > dq["max_dreaddit_train_test_overlap_ratio"]:
            raise ValueError(
                "[dreaddit] train-test clean text overlap "
                f"{overlap_ratio:.2%} exceeds "
                "max_dreaddit_train_test_overlap_ratio="
                f"{dq['max_dreaddit_train_test_overlap_ratio']:.2%}"
            )
        print(f"[DataGate] Dreaddit train-test overlap={overlap_ratio:.2%}")

    return result


if __name__ == "__main__":
    cfg = load_config("config.yaml")
    datasets = prepare_all_datasets(cfg)
    for name, ds in datasets.items():
        print(f"  {name}: {len(ds)} samples | labels={set(ds.labels)}")
