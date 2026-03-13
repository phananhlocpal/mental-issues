"""Node feature computation for the heterogeneous graph.

Features:
  - Document  : SciBERT CLS embedding (768-d) + optional LIWC/DAL (33-d)
  - Word      : GloVe 100-d (or random init) → projected
  - Concept   : BERT definition embedding (768-d) → projected
  - Category  : mean of concept embeddings → projected
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from src.entity_extraction import ALL_CATEGORIES, ALL_CONCEPTS, ConceptEmbedder
from src.graph.graph_builder import HeteroGraphData
from src.preprocessing import DomainDataset


# ──────────────────────────────────────────────────────────────────────────────
# SciBERT document embeddings
# ──────────────────────────────────────────────────────────────────────────────

class DocumentEmbedder:
    """Encode documents with transformer CLS token."""

    def __init__(
        self,
        model_name: str = "allenai/scibert_scivocab_uncased",
        fallback_model_name: str = "bert-base-uncased",
        batch_size: int = 16,
        device: str = "cpu",
    ) -> None:
        self.batch_size = batch_size
        self.device = torch.device(device)
        print(f"  [DocumentEmbedder] Loading {model_name} …")

        # Prefer safetensors weights. If the primary model does not provide
        # safetensors and torch is <2.6, fall back to a safe model.
        self.tokenizer, self.model = self._load_model_with_fallback(
            model_name=model_name,
            fallback_model_name=fallback_model_name,
        )
        self.model.eval()

    def _load_model_with_fallback(
        self,
        model_name: str,
        fallback_model_name: str,
    ) -> tuple[AutoTokenizer, AutoModel]:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name, use_safetensors=True).to(self.device)
            return tokenizer, model
        except Exception as safe_err:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name).to(self.device)
                return tokenizer, model
            except ValueError as raw_err:
                msg = str(raw_err)
                if "upgrade torch to at least v2.6" not in msg:
                    raise

                print(
                    "  [DocumentEmbedder] Current torch cannot load .bin checkpoints safely. "
                    f"Falling back to {fallback_model_name} (safetensors)."
                )
                try:
                    tokenizer = AutoTokenizer.from_pretrained(fallback_model_name)
                    model = AutoModel.from_pretrained(
                        fallback_model_name,
                        use_safetensors=True,
                    ).to(self.device)
                    return tokenizer, model
                except Exception as fallback_err:
                    raise RuntimeError(
                        "Unable to load document encoder safely. "
                        "Upgrade torch to >=2.6, or set preprocessing.safe_fallback_model "
                        "to a model with safetensors."
                    ) from fallback_err
            except Exception:
                raise safe_err

    @torch.no_grad()
    def embed(self, texts: list[str]) -> torch.Tensor:
        all_embs: list[torch.Tensor] = []
        for i in tqdm(
            range(0, len(texts), self.batch_size),
            desc="  Encoding documents",
            leave=False,
        ):
            batch = texts[i : i + self.batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)
            out = self.model(**enc)
            cls = out.last_hidden_state[:, 0, :].cpu()
            all_embs.append(cls)
        return torch.cat(all_embs, dim=0)  # (N_docs, 768)


# ──────────────────────────────────────────────────────────────────────────────
# GloVe word embeddings
# ──────────────────────────────────────────────────────────────────────────────

def load_glove(glove_path: str | Path, dim: int = 100) -> dict[str, np.ndarray]:
    """Load GloVe vectors from file."""
    glove: dict[str, np.ndarray] = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            vec = np.array(parts[1:], dtype=np.float32)
            if len(vec) == dim:
                glove[word] = vec
    return glove


def get_word_embeddings(
    vocab: dict[str, int],
    glove_path: str | None = None,
    dim: int = 100,
) -> torch.Tensor:
    """Return word embeddings matrix (num_words, dim)."""
    num_words = len(vocab)
    emb_matrix = np.random.normal(0, 0.1, (num_words, dim)).astype(np.float32)

    if glove_path and Path(glove_path).exists():
        print(f"  [WordEmbedder] Loading GloVe from {glove_path} …")
        glove = load_glove(glove_path, dim)
        hits = 0
        for word, idx in vocab.items():
            if word in glove:
                emb_matrix[idx] = glove[word]
                hits += 1
        print(f"  GloVe coverage: {hits}/{num_words} ({100*hits/num_words:.1f}%)")
    else:
        print("  [WordEmbedder] GloVe not found – using random init.")

    return torch.tensor(emb_matrix)  # (num_words, dim)


# ──────────────────────────────────────────────────────────────────────────────
# Node feature projection
# ──────────────────────────────────────────────────────────────────────────────

class NodeProjector(nn.Module):
    """Project variable-dim node features to a common projection_dim."""

    def __init__(self, input_dims: dict[str, int], projection_dim: int) -> None:
        super().__init__()
        self.projectors = nn.ModuleDict(
            {
                ntype: nn.Linear(in_dim, projection_dim)
                for ntype, in_dim in input_dims.items()
            }
        )

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {ntype: self.projectors[ntype](feat) for ntype, feat in features.items()}


# ──────────────────────────────────────────────────────────────────────────────
# Full node feature builder
# ──────────────────────────────────────────────────────────────────────────────

def build_node_features(
    graph: HeteroGraphData,
    datasets: list[DomainDataset],
    config: dict,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Build raw (un-projected) node features for all node types."""
    cfg_emb = config["embeddings"]
    cfg_pre = config["preprocessing"]

    # ── Document features (SciBERT) ────────────────────────────────────
    all_raw_texts: list[str] = []
    for ds in datasets:
        all_raw_texts.extend(ds.texts)

    doc_embedder = DocumentEmbedder(
        model_name=cfg_pre.get("scibert_model", "allenai/scibert_scivocab_uncased"),
        fallback_model_name=cfg_pre.get("safe_fallback_model", "bert-base-uncased"),
        device=device,
    )
    doc_feats = doc_embedder.embed(all_raw_texts)  # (N_docs, 768)

    # ── Optional LIWC / DAL feature fusion ────────────────────────────
    if cfg_pre.get("use_liwc_features", False):
        doc_feats = _fuse_liwc_features(doc_feats, datasets)

    # ── Word features (GloVe) ──────────────────────────────────────────
    glove_path = cfg_pre.get("glove_path", None)
    word_feats = get_word_embeddings(
        graph.word_vocab,
        glove_path=glove_path,
        dim=cfg_emb["word_dim"],
    )  # (N_words, 100)

    # ── Concept features (BERT definitions) ───────────────────────────
    concept_embedder = ConceptEmbedder(
        model_name=cfg_pre.get("bert_model", "bert-base-uncased")
    )
    concept_emb_dict = concept_embedder.get_concept_embeddings()
    concept_feats = torch.stack(
        [concept_emb_dict[c] for c in ALL_CONCEPTS]
    )  # (N_concepts, 768)

    # ── Category features (mean of concepts) ──────────────────────────
    cat_emb_dict = concept_embedder.get_category_embeddings(concept_emb_dict)
    category_feats = torch.stack(
        [cat_emb_dict[cat] for cat in ALL_CATEGORIES]
    )  # (N_cats, 768)

    return {
        "document": doc_feats,
        "word": word_feats,
        "medical_concept": concept_feats,
        "symptom_category": category_feats,
    }


def _fuse_liwc_features(
    doc_feats: torch.Tensor,
    datasets: list[DomainDataset],
) -> torch.Tensor:
    """Concatenate normalised LIWC/DAL features to SciBERT document embeddings.

    Datasets with no aux_features (e.g. counseling) are zero-padded to match
    the LIWC dimension of the dreaddit rows.  A StandardScaler is fitted only
    on rows that carry real LIWC data before concatenation.
    """
    from sklearn.preprocessing import StandardScaler  # lazy import

    # Collect raw aux feature arrays
    parts: list[torch.Tensor | None] = []
    liwc_dim: int | None = None
    for ds in datasets:
        mat = getattr(ds, "aux_features", None)
        if mat is not None:
            t = torch.tensor(mat, dtype=torch.float32)
            if liwc_dim is None:
                liwc_dim = t.shape[1]
            parts.append(t)
        else:
            parts.append(None)

    if liwc_dim is None:
        print("  [NodeFeatures] use_liwc_features=true but no aux_features found in any dataset.")
        return doc_feats

    # Fill missing datasets with zeros
    filled: list[torch.Tensor] = [
        p if p is not None else torch.zeros(len(ds), liwc_dim)
        for p, ds in zip(parts, datasets)
    ]
    liwc_feats = torch.cat(filled, dim=0)  # (N_docs, liwc_dim)

    # Fit StandardScaler on rows with real LIWC data (non-zero rows)
    real_mask = torch.tensor([p is not None for p, ds in zip(parts, datasets)
                               for _ in range(len(ds))], dtype=torch.bool)
    liwc_np = liwc_feats.numpy().copy()
    if real_mask.sum() > 0:
        scaler = StandardScaler()
        liwc_np[real_mask.numpy()] = scaler.fit_transform(liwc_np[real_mask.numpy()]).astype(np.float32)
    liwc_feats = torch.tensor(liwc_np, dtype=torch.float32)

    fused = torch.cat([doc_feats, liwc_feats], dim=1)
    print(f"  [NodeFeatures] LIWC fusion: {doc_feats.shape[1]}-d + {liwc_dim}-d → {fused.shape[1]}-d")
    return fused
