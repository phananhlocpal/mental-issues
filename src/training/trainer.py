"""Training loop with domain adversarial learning."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.utils import set_seed


# ──────────────────────────────────────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────────────────────────────────────

class DomainAdversarialLoss(nn.Module):
    """Classification loss + λ * Domain loss."""

    def __init__(self, lam: float = 0.1) -> None:
        super().__init__()
        self.lam = lam
        self.cls_loss_fn = nn.CrossEntropyLoss()
        self.dom_loss_fn = nn.CrossEntropyLoss()

    def forward(
        self,
        cls_logits: torch.Tensor,
        labels: torch.Tensor,
        domain_logits: torch.Tensor | None = None,
        domain_labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        cls_loss = self.cls_loss_fn(cls_logits, labels)
        total = cls_loss
        dom_loss = torch.tensor(0.0, device=cls_logits.device)
        if domain_logits is not None and domain_labels is not None:
            dom_loss = self.dom_loss_fn(domain_logits, domain_labels)
            total = cls_loss + self.lam * dom_loss
        return {"total": total, "cls": cls_loss, "domain": dom_loss}


# ──────────────────────────────────────────────────────────────────────────────
# Trainer
# ──────────────────────────────────────────────────────────────────────────────

class GNNTrainer:
    """Manages training, validation, checkpointing for MentalHealthGNN."""

    def __init__(
        self,
        model: nn.Module,
        config: dict,
        device: torch.device,
        run_name: str = "run",
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.cfg = config
        self.run_name = run_name

        tcfg = config["training"]
        self.epochs = tcfg["epochs"]
        self.lam = tcfg["domain_lambda"]
        self.patience = tcfg["early_stopping_patience"]
        self.lr = tcfg["learning_rate"]

        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.loss_fn = DomainAdversarialLoss(lam=self.lam)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", patience=5, factor=0.5
        )

        ckpt_dir = Path(config["experiments"]["checkpoint_dir"])
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_path = ckpt_dir / f"{run_name}_best.pt"

        log_dir = Path(config["experiments"]["log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / f"{run_name}_log.json"

        self.history: list[dict] = []

    # ------------------------------------------------------------------
    def _forward_batch(
        self,
        pyg_data,
        doc_indices: torch.Tensor,
        labels: torch.Tensor,
        domain_labels: torch.Tensor,
        alpha: float,
    ) -> dict[str, torch.Tensor]:
        out = self.model(
            x_dict=pyg_data.x_dict,
            edge_index_dict=pyg_data.edge_index_dict,
            doc_indices=doc_indices,
            alpha=alpha,
        )
        losses = self.loss_fn(
            cls_logits=out["logits"],
            labels=labels,
            domain_logits=out.get("domain_logits"),
            domain_labels=domain_labels if out.get("domain_logits") is not None else None,
        )
        return {"losses": losses, "logits": out["logits"]}

    # ------------------------------------------------------------------
    def _alpha(self, epoch: int) -> float:
        """Gradually increase domain reversal strength."""
        p = epoch / max(self.epochs, 1)
        return float(2.0 / (1.0 + torch.exp(torch.tensor(-10.0 * p))) - 1.0)

    # ------------------------------------------------------------------
    def train_epoch(
        self,
        pyg_data,
        train_indices: torch.Tensor,
        labels: torch.Tensor,
        domain_labels: torch.Tensor,
        batch_size: int,
        epoch: int,
    ) -> dict[str, float]:
        self.model.train()
        alpha = self._alpha(epoch)
        total_loss = cls_loss = dom_loss = 0.0
        correct = 0

        perm = torch.randperm(len(train_indices))
        for start in range(0, len(train_indices), batch_size):
            idx = perm[start : start + batch_size]
            batch_doc_idx = train_indices[idx].to(self.device)
            batch_labels = labels[batch_doc_idx.cpu()].to(self.device)
            batch_domain = domain_labels[batch_doc_idx.cpu()].to(self.device)

            self.optimizer.zero_grad()
            result = self._forward_batch(
                pyg_data, batch_doc_idx, batch_labels, batch_domain, alpha
            )
            result["losses"]["total"].backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += result["losses"]["total"].item()
            cls_loss += result["losses"]["cls"].item()
            dom_loss += result["losses"]["domain"].item()
            preds = result["logits"].argmax(dim=-1)
            correct += (preds == batch_labels).sum().item()

        n_batches = max(1, len(train_indices) // batch_size)
        n = len(train_indices)
        return {
            "loss": total_loss / n_batches,
            "cls_loss": cls_loss / n_batches,
            "dom_loss": dom_loss / n_batches,
            "acc": correct / n,
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(
        self,
        pyg_data,
        eval_indices: torch.Tensor,
        labels: torch.Tensor,
        domain_labels: torch.Tensor,
        batch_size: int,
    ) -> dict[str, float]:
        self.model.eval()
        all_preds: list[int] = []
        all_labels: list[int] = []
        total_loss = 0.0

        for start in range(0, len(eval_indices), batch_size):
            batch_doc_idx = eval_indices[start : start + batch_size].to(self.device)
            batch_labels = labels[batch_doc_idx.cpu()].to(self.device)
            batch_domain = domain_labels[batch_doc_idx.cpu()].to(self.device)

            out = self.model(
                x_dict=pyg_data.x_dict,
                edge_index_dict=pyg_data.edge_index_dict,
                doc_indices=batch_doc_idx,
                alpha=0.0,
            )
            losses = self.loss_fn(out["logits"], batch_labels)
            total_loss += losses["total"].item()
            preds = out["logits"].argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(batch_labels.cpu().tolist())

        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="binary", zero_division=0)
        prec = precision_score(all_labels, all_preds, average="binary", zero_division=0)
        rec = recall_score(all_labels, all_preds, average="binary", zero_division=0)

        n_batches = max(1, len(eval_indices) // batch_size)
        return {
            "loss": total_loss / n_batches,
            "accuracy": acc,
            "f1": f1,
            "precision": prec,
            "recall": rec,
        }

    # ------------------------------------------------------------------
    def fit(
        self,
        pyg_data,
        train_indices: torch.Tensor,
        val_indices: torch.Tensor,
        labels: torch.Tensor,
        domain_labels: torch.Tensor,
    ) -> list[dict]:
        batch_size = self.cfg["training"]["batch_size"]
        best_f1 = 0.0
        no_improve = 0

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_stats = self.train_epoch(
                pyg_data, train_indices, labels, domain_labels, batch_size, epoch
            )
            val_stats = self.evaluate(
                pyg_data, val_indices, labels, domain_labels, batch_size
            )
            self.scheduler.step(val_stats["f1"])

            elapsed = time.time() - t0
            row = {
                "epoch": epoch,
                "train_loss": round(train_stats["loss"], 4),
                "train_acc": round(train_stats["acc"], 4),
                "val_loss": round(val_stats["loss"], 4),
                "val_acc": round(val_stats["accuracy"], 4),
                "val_f1": round(val_stats["f1"], 4),
                "val_precision": round(val_stats["precision"], 4),
                "val_recall": round(val_stats["recall"], 4),
                "elapsed_s": round(elapsed, 2),
            }
            self.history.append(row)

            print(
                f"  Epoch {epoch:3d}/{self.epochs} | "
                f"loss={train_stats['loss']:.4f} | "
                f"val_f1={val_stats['f1']:.4f} | "
                f"val_acc={val_stats['accuracy']:.4f} | "
                f"{elapsed:.1f}s"
            )

            if val_stats["f1"] > best_f1:
                best_f1 = val_stats["f1"]
                no_improve = 0
                torch.save(self.model.state_dict(), self.ckpt_path)
            else:
                no_improve += 1
                if no_improve >= self.patience:
                    print(f"  Early stopping at epoch {epoch}.")
                    break

        # Save log
        with open(self.log_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"  Training log → {self.log_path}")
        return self.history

    def load_best(self) -> None:
        if self.ckpt_path.exists():
            self.model.load_state_dict(torch.load(self.ckpt_path, map_location=self.device))
            print(f"  Loaded best checkpoint from {self.ckpt_path}")
