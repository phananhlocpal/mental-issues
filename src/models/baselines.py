"""Baseline models for comparison with the HGNN system.

Baselines:
    1. TF-IDF + Logistic Regression   (classical ML)
    2. TF-IDF + Linear SVM            (classical ML)
    3. SciBERT-MLP                    (deep, no graph — uses pre-computed embeddings)
    4. HomoGCN                        (graph, homogeneous GCN)
    5. HomoGAT                        (graph, homogeneous GAT)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# 1–2.  TF-IDF + sklearn classifiers
# ──────────────────────────────────────────────────────────────────────────────

class TFIDFBaseline:
    """Sklearn pipeline: TF-IDF unigram/bigrams + LR or LinearSVC."""

    def __init__(
        self,
        classifier: str = "lr",
        max_features: int = 20_000,
        ngram_range: tuple[int, int] = (1, 2),
    ) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import LinearSVC
        from sklearn.pipeline import Pipeline
        from sklearn.calibration import CalibratedClassifierCV

        if classifier == "lr":
            clf = LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs")
        else:
            base_svc = LinearSVC(max_iter=2000, C=1.0)
            clf = CalibratedClassifierCV(base_svc)          # enables predict_proba

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                sublinear_tf=True,
            )),
            ("clf", clf),
        ])
        self.name = f"TF-IDF + {'LR' if classifier == 'lr' else 'SVM'}"

    def fit(self, texts: list[str], labels: list[int]) -> "TFIDFBaseline":
        self.pipeline.fit(texts, labels)
        return self

    def predict(self, texts: list[str]) -> np.ndarray:
        return self.pipeline.predict(texts)

    def predict_proba(self, texts: list[str]) -> np.ndarray | None:
        clf = self.pipeline.named_steps["clf"]
        if hasattr(clf, "predict_proba"):
            return self.pipeline.predict_proba(texts)[:, 1]
        return None


# ──────────────────────────────────────────────────────────────────────────────
# 3.  SciBERT-MLP: pre-computed [CLS] embeddings → MLP classifier
# ──────────────────────────────────────────────────────────────────────────────

class SciBERTMLPBaseline(nn.Module):
    """MLP on pre-computed SciBERT document embeddings (no graph context)."""

    def __init__(
        self,
        input_dim: int = 768,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        doc_indices: torch.Tensor | None = None,
        **_kwargs,
    ) -> dict[str, torch.Tensor]:
        if doc_indices is not None:
            x = x[doc_indices]
        logits = self.net(x)
        return {"logits": logits, "doc_embeddings": x}


# ──────────────────────────────────────────────────────────────────────────────
# Shared utilities for homo-graph baselines
# ──────────────────────────────────────────────────────────────────────────────

def build_homo_graph(
    pyg_data,
    node_feats: dict[str, torch.Tensor],
    proj_dim: int = 128,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int], dict[str, int]]:
    """Flatten a HeteroData graph into a homogeneous graph for GCN/GAT.

    Returns
    -------
    x : (N_total, proj_dim) — concatenated & linearly-projected features
    edge_index : (2, E_total) — global node indices
    offsets : {node_type: start_index}
    n_nodes  : {node_type: count}
    """
    node_types = pyg_data.node_types

    # ── compute offsets ──────────────────────────────────────────────────────
    offsets: dict[str, int] = {}
    n_nodes: dict[str, int] = {}
    total = 0
    for nt in node_types:
        offsets[nt] = total
        cnt = pyg_data[nt].num_nodes
        n_nodes[nt] = cnt
        total += cnt

    # ── project each node type to proj_dim with a simple linear layer ────────
    proj_layers: dict[str, nn.Linear] = {}
    for nt in node_types:
        in_dim = node_feats[nt].shape[1]
        proj_layers[nt] = nn.Linear(in_dim, proj_dim, bias=False)

    # Initialise all projections and concat
    feat_parts: list[torch.Tensor] = []
    for nt in node_types:
        feat = node_feats[nt].float()
        if feat.shape[1] != proj_dim:
            if feat.shape[1] > proj_dim:
                # Adaptive average pool: works for any input dim
                feat = torch.nn.functional.adaptive_avg_pool1d(
                    feat.unsqueeze(0), proj_dim
                ).squeeze(0)
            else:
                # Zero-pad
                pad = torch.zeros(feat.shape[0], proj_dim - feat.shape[1])
                feat = torch.cat([feat, pad], dim=1)
        feat_parts.append(feat)
    x = torch.cat(feat_parts, dim=0).to(device)

    # ── combine all edges ────────────────────────────────────────────────────
    all_edges: list[torch.Tensor] = []
    for et in pyg_data.edge_types:
        src_type, _rel, dst_type = et
        ei = pyg_data[et].edge_index.to(device)           # (2, E)
        src = ei[0] + offsets[src_type]
        dst = ei[1] + offsets[dst_type]
        all_edges.append(torch.stack([src, dst], dim=0))
    edge_index = torch.cat(all_edges, dim=1)

    return x, edge_index, offsets, n_nodes


# ──────────────────────────────────────────────────────────────────────────────
# 4.  Homogeneous GCN baseline
# ──────────────────────────────────────────────────────────────────────────────

class HomoGCNBaseline(nn.Module):
    """GCN on the flattened (homogeneous) version of the hetero graph."""

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.5,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        try:
            from torch_geometric.nn import GCNConv
        except ImportError:
            raise ImportError("torch-geometric is required for HomoGCNBaseline.")

        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        doc_indices: torch.Tensor | None = None,
        **_kwargs,
    ) -> dict[str, torch.Tensor]:
        for conv in self.convs[:-1]:
            x = F.relu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)

        doc_h = x[doc_indices] if doc_indices is not None else x
        return {"logits": self.classifier(doc_h), "doc_embeddings": doc_h}


# ──────────────────────────────────────────────────────────────────────────────
# 5.  Homogeneous GAT baseline
# ──────────────────────────────────────────────────────────────────────────────

class HomoGATBaseline(nn.Module):
    """GAT on the flattened (homogeneous) version of the hetero graph."""

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 128,
        num_classes: int = 2,
        dropout: float = 0.5,
        num_heads: int = 4,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        try:
            from torch_geometric.nn import GATConv
        except ImportError:
            raise ImportError("torch-geometric is required for HomoGATBaseline.")

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        head_dim = hidden_dim // num_heads

        self.dropout = dropout
        self.convs = nn.ModuleList()
        self.convs.append(GATConv(input_dim, head_dim, heads=num_heads, dropout=dropout))
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_dim, head_dim, heads=num_heads, dropout=dropout))

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        doc_indices: torch.Tensor | None = None,
        **_kwargs,
    ) -> dict[str, torch.Tensor]:
        for conv in self.convs[:-1]:
            x = F.elu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)

        doc_h = x[doc_indices] if doc_indices is not None else x
        return {"logits": self.classifier(doc_h), "doc_embeddings": doc_h}
