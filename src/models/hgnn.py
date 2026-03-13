"""Heterogeneous Graph Attention Network (HGAT) for mental health detection.

Architecture:
    Input node features (projected to hidden_dim)
    → HGATConv layers (type-specific attention)
    → Document node aggregation
    → Classifier head
    → (optional) Domain Adversarial head via Gradient Reversal
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import HGTConv, Linear
    from torch_geometric.data import HeteroData
    _PYG_AVAILABLE = True
except ImportError:
    _PYG_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# Focal Loss (Phase D) – helps with class-imbalanced stress detection
# ──────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Focal loss: down-weights easy examples so the model focuses on hard ones."""

    def __init__(
        self,
        gamma: float = 2.0,
        weight: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)          # probability of the correct class
        fl = (1.0 - pt) ** self.gamma * ce
        return fl.mean() if self.reduction == "mean" else fl.sum()


# ──────────────────────────────────────────────────────────────────────────────
# Gradient Reversal Layer
# ──────────────────────────────────────────────────────────────────────────────

class GradientReversal(torch.autograd.Function):
    """Reverses gradients during backward pass (domain adversarial training)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.alpha * grad_output, None


def grad_reverse(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    return GradientReversal.apply(x, alpha)


# ──────────────────────────────────────────────────────────────────────────────
# Node-type input projector
# ──────────────────────────────────────────────────────────────────────────────

class InputProjection(nn.Module):
    """Project each node type to a common hidden_dim."""

    def __init__(self, input_dims: dict[str, int], hidden_dim: int) -> None:
        super().__init__()
        self.projs = nn.ModuleDict(
            {ntype: nn.Linear(idim, hidden_dim) for ntype, idim in input_dims.items()}
        )

    def forward(self, x_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            ntype: F.relu(self.projs[ntype](x))
            for ntype, x in x_dict.items()
            if ntype in self.projs
        }


# ──────────────────────────────────────────────────────────────────────────────
# HGT-based encoder (requires PyG)
# ──────────────────────────────────────────────────────────────────────────────

class HGTEncoder(nn.Module):
    """Two-layer Heterogeneous Graph Transformer encoder."""

    NODE_TYPES = ["document", "word", "medical_concept", "symptom_category"]
    EDGE_TYPES = [
        ("document", "contains", "word"),
        ("word", "co_occurs", "word"),
        ("word", "maps_to", "medical_concept"),
        ("medical_concept", "belongs_to", "symptom_category"),
        ("medical_concept", "related_to", "medical_concept"),
        # Reverse edges for message passing
        ("word", "rev_contains", "document"),
        ("medical_concept", "rev_maps_to", "word"),
        ("symptom_category", "rev_belongs_to", "medical_concept"),
    ]

    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.5,
        metadata: tuple | None = None,
    ) -> None:
        super().__init__()
        if not _PYG_AVAILABLE:
            raise ImportError("torch-geometric is required for HGTEncoder.")

        if metadata is None:
            # Default metadata from INSTRUCTION
            node_types = self.NODE_TYPES
            edge_types = self.EDGE_TYPES
            metadata = (node_types, edge_types)

        self.dropout = dropout
        self.convs = nn.ModuleList(
            [
                HGTConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    metadata=metadata,
                    heads=num_heads,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict,
    ) -> dict[str, torch.Tensor]:
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {
                ntype: F.dropout(F.relu(x), p=self.dropout, training=self.training)
                for ntype, x in x_dict.items()
            }
        return x_dict


# ──────────────────────────────────────────────────────────────────────────────
# Full model
# ──────────────────────────────────────────────────────────────────────────────

class MentalHealthGNN(nn.Module):
    """Complete HGNN model with optional domain adversarial head.

    When ``use_skip_connection=True`` (default), the raw document features are
    projected *independently* of the message-passing path and concatenated with
    the GNN output before classification.  This residual path ensures that
    informative initial document features (SciBERT + LIWC) are preserved even
    if neighbourhood aggregation over randomly-initialised word nodes would
    otherwise dilute them.
    """

    def __init__(
        self,
        input_dims: dict[str, int],
        hidden_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.5,
        num_classes: int = 2,
        num_domains: int = 2,
        use_domain_adversarial: bool = True,
        use_skip_connection: bool = True,
        metadata: tuple | None = None,
    ) -> None:
        super().__init__()
        self.use_domain_adversarial = use_domain_adversarial
        self.use_skip_connection = use_skip_connection

        self.input_proj = InputProjection(input_dims, hidden_dim)

        if _PYG_AVAILABLE:
            self.encoder = HGTEncoder(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout,
                metadata=metadata,
            )
        else:
            self.encoder = None

        # Skip-connection projection: raw doc features → hidden_dim
        doc_in_dim = input_dims.get("document", hidden_dim)
        if use_skip_connection:
            self.doc_skip_proj = nn.Sequential(
                nn.Linear(doc_in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            classifier_in = 2 * hidden_dim
        else:
            self.doc_skip_proj = None
            classifier_in = hidden_dim

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        # Domain discriminator head
        if use_domain_adversarial:
            self.domain_classifier = nn.Sequential(
                nn.Linear(classifier_in, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_domains),
            )

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict,
        doc_indices: torch.Tensor | None = None,
        alpha: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        # Store raw document features before any projection (for skip path)
        raw_doc_x = x_dict["document"]  # (N_docs, doc_in_dim)

        # Project all node features to hidden_dim
        h = self.input_proj(x_dict)

        # Graph encoding
        if self.encoder is not None:
            h = self.encoder(h, edge_index_dict)

        # Extract document node representations
        doc_h = h["document"]
        if doc_indices is not None:
            doc_h = doc_h[doc_indices]
            raw_doc_x = raw_doc_x[doc_indices]

        # Skip connection: concatenate GNN output with direct projection of raw features
        if self.use_skip_connection and self.doc_skip_proj is not None:
            skip_h = self.doc_skip_proj(raw_doc_x)
            doc_h = torch.cat([doc_h, skip_h], dim=1)  # (N, 2*hidden_dim)

        # Classification
        logits = self.classifier(doc_h)

        out = {"logits": logits, "doc_embeddings": doc_h}

        # Domain adversarial
        if self.use_domain_adversarial and self.training:
            reversed_h = grad_reverse(doc_h, alpha)
            domain_logits = self.domain_classifier(reversed_h)
            out["domain_logits"] = domain_logits

        return out


# ──────────────────────────────────────────────────────────────────────────────
# Simple BERT-only baseline
# ──────────────────────────────────────────────────────────────────────────────

class BERTBaseline(nn.Module):
    """Vanilla BERT fine-tuning baseline for comparison."""

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_classes: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(model_name)
        hidden = self.bert.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(cls)
