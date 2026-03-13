"""Trainer utilities for baseline models (sklearn + PyTorch MLP/GCN/GAT)."""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.models.baselines import TFIDFBaseline


# ──────────────────────────────────────────────────────────────────────────────
# sklearn baseline runner (TF-IDF + LR / SVM)
# ──────────────────────────────────────────────────────────────────────────────

def run_sklearn_baseline(
    baseline: TFIDFBaseline,
    train_texts: list[str],
    train_labels: list[int],
    test_texts: list[str],
    test_labels: list[int],
) -> dict[str, Any]:
    """Train & evaluate a TFIDFBaseline, return metrics dict."""
    t0 = time.time()
    baseline.fit(train_texts, train_labels)
    elapsed = time.time() - t0

    preds = baseline.predict(test_texts)
    metrics = {
        "accuracy":  accuracy_score(test_labels, preds),
        "precision": precision_score(test_labels, preds, average="binary", zero_division=0),
        "recall":    recall_score(test_labels, preds, average="binary", zero_division=0),
        "f1":        f1_score(test_labels, preds, average="binary", zero_division=0),
        "train_time_s": round(elapsed, 2),
    }
    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# Embedding-based trainer (SciBERT-MLP, GCN, GAT)
# ──────────────────────────────────────────────────────────────────────────────

class EmbeddingTrainer:
    """Generic trainer for PyTorch baselines that work on pre-built tensors.

    Supports two graph modes:
      - mode="mlp"  : inputs are (x_doc, doc_indices) — no graph needed
      - mode="gnn"  : inputs are (x_all, edge_index, doc_indices)
    """

    def __init__(
        self,
        model: nn.Module,
        mode: str = "mlp",          # "mlp" | "gnn"
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 64,
        patience: int = 10,
        device: torch.device | str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.mode = mode
        self.device = torch.device(device) if isinstance(device, str) else device
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience

        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", patience=5, factor=0.5
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.best_state: dict | None = None

    # ------------------------------------------------------------------
    def _forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None,
        doc_indices: torch.Tensor,
    ) -> torch.Tensor:
        if self.mode == "mlp":
            out = self.model(x, doc_indices=doc_indices)
        else:
            out = self.model(x, edge_index, doc_indices=doc_indices)
        return out["logits"]

    # ------------------------------------------------------------------
    def _epoch(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None,
        indices: torch.Tensor,
        labels: torch.Tensor,
        train: bool,
    ) -> tuple[float, float]:
        """Run one epoch; return (avg_loss, accuracy)."""
        self.model.train(train)
        total_loss = correct = 0
        perm = torch.randperm(len(indices)) if train else torch.arange(len(indices))
        n_batches = 0

        for start in range(0, len(perm), self.batch_size):
            batch_rel = perm[start : start + self.batch_size]
            batch_idx = indices[batch_rel].to(self.device)
            batch_lbl = labels[batch_idx.cpu()].to(self.device)

            ctx = torch.enable_grad() if train else torch.no_grad()
            with ctx:
                logits = self._forward(x, edge_index, batch_idx)
                loss = self.loss_fn(logits, batch_lbl)

            if train:
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(-1) == batch_lbl).sum().item()
            n_batches += 1

        n_batches = max(1, n_batches)
        return total_loss / n_batches, correct / len(indices)

    # ------------------------------------------------------------------
    def fit(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None,
        train_indices: torch.Tensor,
        val_indices: torch.Tensor,
        labels: torch.Tensor,
    ) -> "EmbeddingTrainer":
        x = x.to(self.device)
        if edge_index is not None:
            edge_index = edge_index.to(self.device)

        best_f1 = 0.0
        no_improve = 0

        for epoch in range(1, self.epochs + 1):
            tr_loss, tr_acc = self._epoch(x, edge_index, train_indices, labels, train=True)
            val_metrics = self.evaluate(x, edge_index, val_indices, labels)
            self.scheduler.step(val_metrics["f1"])

            if val_metrics["f1"] > best_f1:
                best_f1 = val_metrics["f1"]
                no_improve = 0
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    break

        if self.best_state is not None:
            self.model.load_state_dict(
                {k: v.to(self.device) for k, v in self.best_state.items()}
            )
        return self

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor | None,
        indices: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, float]:
        self.model.eval()
        x = x.to(self.device)
        if edge_index is not None:
            edge_index = edge_index.to(self.device)

        all_preds: list[int] = []
        all_true: list[int] = []
        total_loss = 0.0
        n_batches = 0

        for start in range(0, len(indices), self.batch_size):
            batch_idx = indices[start : start + self.batch_size].to(self.device)
            batch_lbl = labels[batch_idx.cpu()].to(self.device)

            logits = self._forward(x, edge_index, batch_idx)
            total_loss += self.loss_fn(logits, batch_lbl).item()
            n_batches += 1
            all_preds.extend(logits.argmax(-1).cpu().tolist())
            all_true.extend(batch_lbl.cpu().tolist())

        return {
            "loss":      total_loss / max(n_batches, 1),
            "accuracy":  accuracy_score(all_true, all_preds),
            "precision": precision_score(all_true, all_preds, average="binary", zero_division=0),
            "recall":    recall_score(all_true, all_preds, average="binary", zero_division=0),
            "f1":        f1_score(all_true, all_preds, average="binary", zero_division=0),
        }
