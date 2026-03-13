"""Phase D smoke test – validates all Phase D additions."""
import torch
import yaml
from src.models import FocalLoss
from src.entity_extraction import CONCEPT_RELATIONS, Entity

# ------------------------------------------------------------------
# FocalLoss
# ------------------------------------------------------------------
fl = FocalLoss(gamma=2.0)
logits = torch.randn(8, 2)
targets = torch.randint(0, 2, (8,))
loss = fl(logits, targets)
assert loss.item() >= 0, "FocalLoss negative"
print(f"FocalLoss OK: {round(loss.item(), 4)}")

# Weighted focal loss
weights = torch.tensor([1.0, 2.0])
fl_w = FocalLoss(gamma=2.0, weight=weights)
loss_w = fl_w(logits, targets)
assert loss_w.item() >= 0, "FocalLoss weighted negative"
print(f"FocalLoss (weighted) OK: {round(loss_w.item(), 4)}")

# ------------------------------------------------------------------
# Entity confidence field
# ------------------------------------------------------------------
e = Entity(surface="anxiety", category="anxiety_disorder", concept="anxiety")
assert e.confidence == 1.0, f"Expected 1.0 got {e.confidence}"
print(f"Entity default confidence: {e.confidence}  OK")

e2 = Entity(
    surface="chronic anxiety",
    category="anxiety_disorder",
    concept="anxiety",
    confidence=0.85,
)
assert e2.confidence == 0.85, f"Expected 0.85 got {e2.confidence}"
print(f"Entity explicit confidence: {e2.confidence}  OK")

# ------------------------------------------------------------------
# CONCEPT_RELATIONS
# ------------------------------------------------------------------
assert len(CONCEPT_RELATIONS) >= 10, "Too few CONCEPT_RELATIONS"
print(f"CONCEPT_RELATIONS count: {len(CONCEPT_RELATIONS)}  OK")

# ------------------------------------------------------------------
# Config Phase D keys
# ------------------------------------------------------------------
cfg = yaml.safe_load(open("config.yaml"))
g = cfg["graph"]
t = cfg["training"]
assert g["min_cooc_count"] == 2
assert g["normalize_doc_word_weights"] is True
assert g["min_entity_confidence"] == 0.8
assert t["use_focal_loss"] is False
assert t["focal_gamma"] == 2.0
print(f"config graph.min_cooc_count: {g['min_cooc_count']}  OK")
print(f"config graph.normalize_doc_word_weights: {g['normalize_doc_word_weights']}  OK")
print(f"config graph.min_entity_confidence: {g['min_entity_confidence']}  OK")
print(f"config training.use_focal_loss: {t['use_focal_loss']}  OK")
print(f"config training.focal_gamma: {t['focal_gamma']}  OK")

print("\nPhase D smoke test PASSED")
