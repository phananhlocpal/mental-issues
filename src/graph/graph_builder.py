"""Heterogeneous graph construction for mental health detection.

Node types:
    0 – Document
    1 – Word
    2 – Medical Concept
    3 – Symptom Category

Edge types (as string keys):
    ('document', 'contains',    'word')
    ('word',     'co_occurs',   'word')
    ('word',     'maps_to',     'medical_concept')
    ('medical_concept', 'belongs_to', 'symptom_category')
    ('medical_concept', 'related_to', 'medical_concept')
"""
from __future__ import annotations

import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from src.entity_extraction import (
    ALL_CATEGORIES,
    ALL_CONCEPTS,
    CONCEPT_RELATIONS,
    LEXICON,
    MedicalEntityExtractor,
)
from src.preprocessing import DomainDataset


# ──────────────────────────────────────────────────────────────────────────────
# Graph data container
# ──────────────────────────────────────────────────────────────────────────────

class HeteroGraphData:
    """Lightweight heterogeneous graph without PyG dependency at construction time."""

    def __init__(self) -> None:
        # Node index maps
        self.doc_ids: list[int] = []          # original sample indices
        self.word_vocab: dict[str, int] = {}  # word → word_node_idx
        self.concept_vocab: dict[str, int] = {}  # concept → concept_node_idx
        self.category_vocab: dict[str, int] = {}  # category → cat_node_idx

        # Edge tensors  (src, dst, weight)
        self.doc_word_edges: list[tuple[int, int, float]] = []
        self.word_word_edges: list[tuple[int, int, float]] = []
        self.word_concept_edges: list[tuple[int, int, float]] = []
        self.concept_category_edges: list[tuple[int, int, float]] = []
        self.concept_concept_edges: list[tuple[int, int, float]] = []
        # Semantic relation type per concept-concept edge (Phase D)
        self.concept_concept_relations: list[str] = []

        # Labels / domain per document node
        self.labels: list[int] = []
        self.domain_ids: list[int] = []

    # ------------------------------------------------------------------
    @property
    def num_docs(self) -> int:
        return len(self.doc_ids)

    @property
    def num_words(self) -> int:
        return len(self.word_vocab)

    @property
    def num_concepts(self) -> int:
        return len(self.concept_vocab)

    @property
    def num_categories(self) -> int:
        return len(self.category_vocab)

    def summary(self) -> str:
        return (
            f"Nodes → docs:{self.num_docs}  words:{self.num_words}  "
            f"concepts:{self.num_concepts}  categories:{self.num_categories}\n"
            f"Edges → doc-word:{len(self.doc_word_edges)}  "
            f"word-word:{len(self.word_word_edges)}  "
            f"word-concept:{len(self.word_concept_edges)}  "
            f"concept-cat:{len(self.concept_category_edges)}  "
            f"concept-concept:{len(self.concept_concept_edges)}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Builder
# ──────────────────────────────────────────────────────────────────────────────

class HeteroGraphBuilder:
    """Build a HeteroGraphData from one or more DomainDataset objects."""

    def __init__(self, config: dict) -> None:
        self.cfg = config
        self.window = config["preprocessing"]["co_occurrence_window"]
        self.tfidf_top_k = config["preprocessing"]["tfidf_top_k"]
        graph_cfg = config.get("graph", {})
        # Minimum co-occurrence count to keep a word-word edge (sparsity filter)
        self.min_cooc_count: int = int(graph_cfg.get("min_cooc_count", 2))
        # Whether to L2-normalize doc-word edge weights per document row
        self.normalize_doc_word: bool = bool(graph_cfg.get("normalize_doc_word_weights", True))
        # Minimum entity confidence to create a word-concept edge
        self.min_entity_confidence: float = float(graph_cfg.get("min_entity_confidence", 0.8))
        self.extractor = MedicalEntityExtractor()

    # ------------------------------------------------------------------
    def build(self, datasets: list[DomainDataset]) -> HeteroGraphData:
        g = HeteroGraphData()

        # ── 1. Vocabulary of all words across datasets ──────────────────
        print("[Graph] Building vocabulary …")
        word_freq: Counter = Counter()
        all_token_lists: list[list[str]] = []
        all_clean_texts: list[str] = []
        all_labels: list[int] = []
        all_domain_ids: list[int] = []

        for ds in datasets:
            for tokens, label, dom_id, ct in zip(
                ds.tokens, ds.labels, ds.domain_ids, ds.clean_texts
            ):
                word_freq.update(tokens)
                all_token_lists.append(tokens)
                all_clean_texts.append(ct)
                all_labels.append(label)
                all_domain_ids.append(dom_id)

        # Keep words with freq ≥ 2, then re-index from 0 (no gaps)
        filtered_words = [w for w, c in word_freq.items() if c >= 2]
        vocab = {w: i for i, w in enumerate(filtered_words)}
        g.word_vocab = vocab

        # ── 2. Concept & category vocabs ────────────────────────────────
        g.concept_vocab = {c: i for i, c in enumerate(ALL_CONCEPTS)}
        g.category_vocab = {cat: i for i, cat in enumerate(ALL_CATEGORIES)}

        # ── 3. TF-IDF for doc-word edges ────────────────────────────────
        print("[Graph] Computing TF-IDF doc-word edges …")
        corpus = [" ".join(toks) for toks in all_token_lists]
        tfidf = TfidfVectorizer(
            vocabulary=vocab,
            max_features=len(vocab),
            sublinear_tf=True,
        )
        tfidf_matrix = tfidf.fit_transform(corpus)  # (N_docs, N_words)

        for doc_idx in tqdm(range(len(corpus)), desc="  doc-word edges", leave=False):
            g.doc_ids.append(doc_idx)
            g.labels.append(all_labels[doc_idx])
            g.domain_ids.append(all_domain_ids[doc_idx])
            row = tfidf_matrix[doc_idx]
            _, word_indices = row.nonzero()
            if len(word_indices) == 0:
                continue
            # Keep top-k by TF-IDF score
            scores = np.asarray(row[:, word_indices].todense()).flatten()
            top_k = min(self.tfidf_top_k, len(scores))
            top_idx = np.argsort(scores)[-top_k:]
            raw_weights = np.array([float(tfidf_matrix[doc_idx, word_indices[i]]) for i in top_idx])
            if self.normalize_doc_word and raw_weights.max() > 0:
                raw_weights = raw_weights / raw_weights.max()
            for rank, wi in enumerate(word_indices[top_idx]):
                g.doc_word_edges.append((doc_idx, int(wi), float(raw_weights[rank])))

        # ── 4. Word-word co-occurrence edges ────────────────────────────
        print("[Graph] Building word-word co-occurrence edges …")
        cooc: Counter = Counter()
        for tokens in tqdm(all_token_lists, desc="  co-occurrence", leave=False):
            token_ids = [vocab[t] for t in tokens if t in vocab]
            for i, wi in enumerate(token_ids):
                for wj in token_ids[i + 1 : i + 1 + self.window]:
                    key = (min(wi, wj), max(wi, wj))
                    cooc[key] += 1

        max_cooc = max((v for v in cooc.values()), default=1)
        for (wi, wj), cnt in cooc.items():
            if cnt < self.min_cooc_count:
                continue
            # Normalize count by max to [0,1] and add both directions
            w = float(cnt) / float(max_cooc)
            g.word_word_edges.append((wi, wj, w))
            g.word_word_edges.append((wj, wi, w))

        # ── 5. Word-concept edges ────────────────────────────────────────
        print("[Graph] Building word-concept edges …")
        wc_edges: set[tuple[int, int]] = set()

        # Prefer corpus-driven extraction so graph edges reflect observed entities.
        # Phase D: filter by entity confidence and weight edge by confidence.
        extraction_results = self.extractor.extract_batch(all_clean_texts)
        wc_weighted: dict[tuple[int, int], float] = {}
        for res in extraction_results:
            for ent in res.entities:
                if ent.concept not in g.concept_vocab:
                    continue
                if ent.confidence < self.min_entity_confidence:
                    continue
                ci = g.concept_vocab[ent.concept]
                for tok in ent.surface.split():
                    if tok in vocab:
                        key = (vocab[tok], ci)
                        # take max confidence across all mentions
                        wc_weighted[key] = max(wc_weighted.get(key, 0.0), ent.confidence)
                        wc_edges.add(key)

        # Fallback to lexicon priors if extraction is sparse.
        if not wc_edges:
            for surface, (concept, _) in LEXICON.items():
                if concept not in g.concept_vocab:
                    continue
                ci = g.concept_vocab[concept]
                for tok in surface.split():
                    if tok in vocab:
                        wc_edges.add((vocab[tok], ci))

        g.word_concept_edges = [
            (wi, ci, wc_weighted.get((wi, ci), 1.0)) for wi, ci in sorted(wc_edges)
        ]

        # ── 6. Concept-category edges ────────────────────────────────────
        print("[Graph] Building concept-category edges …")
        for surface, (concept, category) in LEXICON.items():
            ci = g.concept_vocab[concept]
            cati = g.category_vocab[category]
            edge = (ci, cati, 1.0)
            if edge not in g.concept_category_edges:
                g.concept_category_edges.append(edge)

        # ── 7. Concept-concept edges (typed semantic relations, Phase D) ───
        print("[Graph] Building concept-concept edges (semantic relations) …")
        added_pairs: set[tuple[int, int]] = set()

        # Priority 1: explicit semantic relations from CONCEPT_RELATIONS
        for (concept_a, concept_b), rel_type in CONCEPT_RELATIONS.items():
            if concept_a not in g.concept_vocab or concept_b not in g.concept_vocab:
                continue
            ci = g.concept_vocab[concept_a]
            cj = g.concept_vocab[concept_b]
            # directional edge a→b with weight=1.0; mark both directions as seen
            g.concept_concept_edges.append((ci, cj, 1.0))
            g.concept_concept_relations.append(rel_type)
            added_pairs.add((ci, cj))

        # Priority 2: same-category fallback for pairs not in explicit relations
        cat_to_concepts: dict[str, list[int]] = defaultdict(list)
        for surface, (concept, category) in LEXICON.items():
            cat_to_concepts[category].append(g.concept_vocab[concept])

        for cat, concept_indices in cat_to_concepts.items():
            unique_ci = list(set(concept_indices))
            for i, ci in enumerate(unique_ci):
                for cj in unique_ci[i + 1:]:
                    if (ci, cj) not in added_pairs:
                        g.concept_concept_edges.append((ci, cj, 0.5))
                        g.concept_concept_relations.append("same_category")
                        added_pairs.add((ci, cj))

        print("[Graph] Done!\n" + g.summary())
        return g

    # ------------------------------------------------------------------
    def to_pyg(self, g: HeteroGraphData, device: torch.device | None = None):
        """Convert HeteroGraphData to PyG HeteroData object."""
        try:
            from torch_geometric.data import HeteroData
        except ImportError:
            raise ImportError("torch-geometric is required for PyG conversion.")

        data = HeteroData()

        # ── Node feature placeholders (filled by embedding module) ──────
        data["document"].num_nodes = g.num_docs
        data["document"].y = torch.tensor(g.labels, dtype=torch.long)
        data["document"].domain = torch.tensor(g.domain_ids, dtype=torch.long)

        data["word"].num_nodes = g.num_words
        data["medical_concept"].num_nodes = g.num_concepts
        data["symptom_category"].num_nodes = g.num_categories

        # ── Edges ────────────────────────────────────────────────────────
        def make_edge(edges: list[tuple[int, int, float]]):
            if not edges:
                return torch.zeros(2, 0, dtype=torch.long), torch.zeros(0)
            src = torch.tensor([e[0] for e in edges], dtype=torch.long)
            dst = torch.tensor([e[1] for e in edges], dtype=torch.long)
            wt = torch.tensor([e[2] for e in edges], dtype=torch.float)
            return torch.stack([src, dst]), wt

        data["document", "contains", "word"].edge_index, \
            data["document", "contains", "word"].edge_weight = make_edge(g.doc_word_edges)

        data["word", "co_occurs", "word"].edge_index, \
            data["word", "co_occurs", "word"].edge_weight = make_edge(g.word_word_edges)

        data["word", "maps_to", "medical_concept"].edge_index, \
            data["word", "maps_to", "medical_concept"].edge_weight = make_edge(g.word_concept_edges)

        data["medical_concept", "belongs_to", "symptom_category"].edge_index, \
            data["medical_concept", "belongs_to", "symptom_category"].edge_weight = make_edge(
                g.concept_category_edges
            )

        data["medical_concept", "related_to", "medical_concept"].edge_index, \
            data["medical_concept", "related_to", "medical_concept"].edge_weight = make_edge(
                g.concept_concept_edges
            )

        if device:
            data = data.to(device)
        return data


# ──────────────────────────────────────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────────────────────────────────────

def save_graph(graph: HeteroGraphData, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(graph, f)
    print(f"  Graph saved → {path}")


def load_graph(path: str | Path) -> HeteroGraphData:
    with open(path, "rb") as f:
        return pickle.load(f)
