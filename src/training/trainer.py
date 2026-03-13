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
from src.models import FocalLoss


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
            # Domain adversarial signal is meaningful only when at least two domains
            # are present in the batch.
            if domain_labels.unique().numel() > 1:
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
        self.use_full_graph_single_pass = tcfg.get("use_full_graph_single_pass", True)
        # Phase D: optional focal loss for class-imbalanced datasets
        self.use_focal_loss: bool = bool(tcfg.get("use_focal_loss", False))
        self.focal_gamma: float = float(tcfg.get("focal_gamma", 2.0))

        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.loss_fn = DomainAdversarialLoss(lam=self.lam)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", patience=5, factor=0.5
        )
        self.use_class_weight = tcfg.get("use_class_weight", True)

        ckpt_dir = Path(config["experiments"]["checkpoint_dir"])
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_path = ckpt_dir / f"{run_name}_best.pt"

        log_dir = Path(config["experiments"]["log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = log_dir / f"{run_name}_log.json"

        self.history: list[dict] = []

    # ------------------------------------------------------------------
    def _configure_classification_loss(self, train_labels: torch.Tensor) -> None:
        """Set CE or Focal loss, optionally weighted by class frequency."""
        counts = torch.bincount(train_labels.long().cpu(), minlength=2).float()
        valid_counts = (counts > 0).all()

        if self.use_class_weight and valid_counts:
            total = counts.sum()
            weights = (total / (counts.numel() * counts)).to(self.device)
        else:
            weights = None

        if self.use_focal_loss:
            self.loss_fn.cls_loss_fn = FocalLoss(gamma=self.focal_gamma, weight=weights)
        elif weights is not None:
            self.loss_fn.cls_loss_fn = nn.CrossEntropyLoss(weight=weights)
        else:
            self.loss_fn.cls_loss_fn = nn.CrossEntropyLoss()

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

        if self.use_full_graph_single_pass:
            batch_doc_idx = train_indices.to(self.device)
            batch_labels = labels[train_indices.cpu()].to(self.device)
            batch_domain = domain_labels[train_indices.cpu()].to(self.device)

            self.optimizer.zero_grad()
            result = self._forward_batch(
                pyg_data, batch_doc_idx, batch_labels, batch_domain, alpha
            )
            result["losses"]["total"].backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            preds = result["logits"].argmax(dim=-1)
            correct = (preds == batch_labels).sum().item()
            n = len(train_indices)
            return {
                "loss": float(result["losses"]["total"].item()),
                "cls_loss": float(result["losses"]["cls"].item()),
                "dom_loss": float(result["losses"]["domain"].item()),
                "acc": correct / max(1, n),
            }

        total_loss = cls_loss = dom_loss = 0.0
        correct = 0
        n_batches = 0

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
            n_batches += 1

        n_batches = max(1, n_batches)
        n = len(train_indices)
        return {
            "loss": total_loss / n_batches,
            "cls_loss": cls_loss / n_batches,
            "dom_loss": dom_loss / n_batches,
            "acc": correct / n,
        }

    # ------------------------------------------------------------------
    def train_epoch_domain_adversarial(
        self,
        pyg_data,
        source_indices: torch.Tensor,
        target_indices: torch.Tensor,
        labels: torch.Tensor,
        domain_labels: torch.Tensor,
        batch_size: int,
        epoch: int,
    ) -> dict[str, float]:
        """Train one epoch with source-labeled + target-unlabeled batches.

        - Classification loss is computed on source samples only.
        - Domain loss is computed on both source and target samples.
        """
        if len(target_indices) == 0:
            return self.train_epoch(
                pyg_data,
                source_indices,
                labels,
                domain_labels,
                batch_size,
                epoch,
            )

        self.model.train()
        alpha = self._alpha(epoch)

        total_loss = cls_loss_total = dom_loss_total = 0.0
        correct = 0
        n_steps = 0

        # Use enough steps to sweep through the larger side once.
        steps_src = (len(source_indices) + batch_size - 1) // batch_size
        steps_tgt = (len(target_indices) + batch_size - 1) // batch_size
        steps = max(1, steps_src, steps_tgt)

        src_perm = torch.randperm(len(source_indices))
        tgt_perm = torch.randperm(len(target_indices))

        for step in range(steps):
            s0 = (step * batch_size) % len(source_indices)
            t0 = (step * batch_size) % len(target_indices)

            src_rel = src_perm[s0 : s0 + batch_size]
            tgt_rel = tgt_perm[t0 : t0 + batch_size]
            if len(src_rel) == 0:
                continue
            if len(tgt_rel) == 0:
                tgt_rel = tgt_perm[: min(batch_size, len(tgt_perm))]

            src_idx = source_indices[src_rel].to(self.device)
            tgt_idx = target_indices[tgt_rel].to(self.device)

            src_labels = labels[src_idx.cpu()].to(self.device)
            src_domain = domain_labels[src_idx.cpu()].to(self.device)
            tgt_domain = domain_labels[tgt_idx.cpu()].to(self.device)

            self.optimizer.zero_grad()

            out_src = self.model(
                x_dict=pyg_data.x_dict,
                edge_index_dict=pyg_data.edge_index_dict,
                doc_indices=src_idx,
                alpha=alpha,
            )
            out_tgt = self.model(
                x_dict=pyg_data.x_dict,
                edge_index_dict=pyg_data.edge_index_dict,
                doc_indices=tgt_idx,
                alpha=alpha,
            )

            cls_loss = self.loss_fn.cls_loss_fn(out_src["logits"], src_labels)

            dom_loss = torch.tensor(0.0, device=self.device)
            if (
                out_src.get("domain_logits") is not None
                and out_tgt.get("domain_logits") is not None
            ):
                dom_logits = torch.cat(
                    [out_src["domain_logits"], out_tgt["domain_logits"]], dim=0
                )
                dom_targets = torch.cat([src_domain, tgt_domain], dim=0)
                if dom_targets.unique().numel() > 1:
                    dom_loss = self.loss_fn.dom_loss_fn(dom_logits, dom_targets)

            total = cls_loss + self.lam * dom_loss
            total.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            preds = out_src["logits"].argmax(dim=-1)
            correct += (preds == src_labels).sum().item()

            total_loss += total.item()
            cls_loss_total += cls_loss.item()
            dom_loss_total += dom_loss.item()
            n_steps += 1

        n_steps = max(1, n_steps)
        return {
            "loss": total_loss / n_steps,
            "cls_loss": cls_loss_total / n_steps,
            "dom_loss": dom_loss_total / n_steps,
            "acc": correct / max(1, len(source_indices)),
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

        from src.evaluation.metrics import compute_metrics

        if self.use_full_graph_single_pass:
            batch_doc_idx = eval_indices.to(self.device)
            batch_labels = labels[eval_indices.cpu()].to(self.device)

            out = self.model(
                x_dict=pyg_data.x_dict,
                edge_index_dict=pyg_data.edge_index_dict,
                doc_indices=batch_doc_idx,
                alpha=0.0,
            )
            losses = self.loss_fn(out["logits"], batch_labels)
            probs = torch.softmax(out["logits"], dim=-1)[:, 1].cpu().tolist()
            preds = out["logits"].argmax(dim=-1).cpu().tolist()
            y_true = batch_labels.cpu().tolist()
            m = compute_metrics(y_true, preds, probs)
            m["loss"] = float(losses["total"].item())
            return m

        all_preds: list[int] = []
        all_probs: list[float] = []
        all_labels: list[int] = []
        total_loss = 0.0
        n_batches = 0

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
            all_probs.extend(torch.softmax(out["logits"], dim=-1)[:, 1].cpu().tolist())
            all_preds.extend(out["logits"].argmax(dim=-1).cpu().tolist())
            all_labels.extend(batch_labels.cpu().tolist())
            n_batches += 1

        m = compute_metrics(all_labels, all_preds, all_probs)
        m["loss"] = total_loss / max(n_batches, 1)
        return m

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
        train_labels = labels[train_indices.cpu()]
        self._configure_classification_loss(train_labels)
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
                "domain_lambda": self.lam,
                "use_focal_loss": self.use_focal_loss,
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

    # ------------------------------------------------------------------
    def fit_domain_adversarial(
        self,
        pyg_data,
        source_train_indices: torch.Tensor,
        source_val_indices: torch.Tensor,
        target_train_indices: torch.Tensor,
        labels: torch.Tensor,
        domain_labels: torch.Tensor,
    ) -> list[dict]:
        """Fit with unsupervised domain adaptation.

        Source labels are used for classification; target labels are not used.
        """
        batch_size = self.cfg["training"]["batch_size"]
        train_labels = labels[source_train_indices.cpu()]
        self._configure_classification_loss(train_labels)
        best_f1 = 0.0
        no_improve = 0

        for epoch in range(1, self.epochs + 1):
            t0 = time.time()
            train_stats = self.train_epoch_domain_adversarial(
                pyg_data,
                source_indices=source_train_indices,
                target_indices=target_train_indices,
                labels=labels,
                domain_labels=domain_labels,
                batch_size=batch_size,
                epoch=epoch,
            )
            val_stats = self.evaluate(
                pyg_data, source_val_indices, labels, domain_labels, batch_size
            )
            self.scheduler.step(val_stats["f1"])

            elapsed = time.time() - t0
            row = {
                "epoch": epoch,
                "train_loss": round(train_stats["loss"], 4),
                "train_cls_loss": round(train_stats["cls_loss"], 4),
                "train_dom_loss": round(train_stats["dom_loss"], 4),
                "train_acc": round(train_stats["acc"], 4),
                "val_loss": round(val_stats["loss"], 4),
                "val_acc": round(val_stats["accuracy"], 4),
                "val_f1": round(val_stats["f1"], 4),
                "val_precision": round(val_stats["precision"], 4),
                "val_recall": round(val_stats["recall"], 4),
                "elapsed_s": round(elapsed, 2),
                "domain_lambda": self.lam,
                "use_focal_loss": self.use_focal_loss,
            }
            self.history.append(row)

            print(
                f"  Epoch {epoch:3d}/{self.epochs} | "
                f"loss={train_stats['loss']:.4f} | "
                f"cls={train_stats['cls_loss']:.4f} | "
                f"dom={train_stats['dom_loss']:.4f} | "
                f"val_f1={val_stats['f1']:.4f} | "
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

        with open(self.log_path, "w") as f:
            json.dump(self.history, f, indent=2)
        print(f"  Training log → {self.log_path}")
        return self.history

    def load_best(self) -> None:
        if self.ckpt_path.exists():
            self.model.load_state_dict(torch.load(self.ckpt_path, map_location=self.device))
            print(f"  Loaded best checkpoint from {self.ckpt_path}")
