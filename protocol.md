# Experimental Protocol (Train/Val/Test)

## Scope
This protocol defines the only accepted evaluation setup for in-domain stress detection and a guarded setup for cross-domain adaptation.

## Dataset Policy
1. In-domain benchmark uses Dreaddit only.
2. Train/Val come from `dreaddit_train.csv` via stratified split.
3. Test uses `dreaddit_test.csv` only and is never used for model selection.
4. Any dataset used for supervised classification must pass data-quality gates.
5. Any single-class dataset is disallowed for supervised target evaluation unless explicitly marked as adaptation-only.

## Data-Quality Gates
Applied at load time:
1. Empty clean-text ratio must be <= `data_quality.max_empty_ratio`.
2. Duplicate clean-text ratio must be <= `data_quality.max_duplicate_ratio`.
3. Required datasets in `data_quality.require_binary_labels_for` must have labels in `{0, 1}`.
4. Single-class datasets are only allowed if listed in `data_quality.allow_single_class_for`.
5. Optional de-duplication can be enabled with `data_quality.deduplicate_clean_text`.

## In-Domain Split (Official)
1. Load both `dreaddit_train` and `dreaddit_test`.
2. Build stratified indices on `dreaddit_train` only:
- `train_idx`
- `val_idx`
3. Define `test_idx = [0..len(dreaddit_test)-1]`.
4. Tune hyperparameters and select checkpoints only on validation metrics.
5. Run final evaluation once on official test split.

Implementation helper:
- `build_dreaddit_protocol_splits(...)` in [src/preprocessing/data_loader.py](src/preprocessing/data_loader.py).

## Cross-Domain Rules
1. Cross-domain training is allowed only if source supervised set has >=2 classes.
2. Target test split must be hold-out and excluded from adaptation and model selection.
3. If target dataset is single-class, report only adaptation diagnostics, not supervised target F1/accuracy claims.

## Leakage Control
1. Feature fitting must not use test labels.
2. If transductive graph mode is used (test nodes present in graph), this must be explicitly reported and compared with a non-transductive baseline.
3. Official headline metric must come from the predefined in-domain protocol.

## Reporting Requirements
1. Report `accuracy`, `macro_f1`, per-class precision/recall/F1.
2. Report mean and std across multiple seeds.
3. Keep raw logs and split artifacts under `experiments/logs` and `experiments/results`.

## Reproducibility Checklist
1. Fixed random seed in config.
2. Version-locked dependencies.
3. Saved split indices for each run.
4. Saved best checkpoint with run metadata.
