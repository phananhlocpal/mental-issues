# Dataset Card (Internal)

## 1. Sources
- dreaddit_train: `data/raw/dreaddit_train.csv`
- dreaddit_test: `data/raw/dreaddit_test.csv`
- counseling: `data/raw/counseling.csv` (from archive canonical `Dataset.csv`)

## 2. Task Mapping
- Primary supervised task: binary stress detection on Dreaddit (`label` in {0,1}).
- Counseling currently used as adaptation-only unlabeled domain in cross-domain settings.

## 3. Split Protocol
- In-domain:
  - Train: `dreaddit_train`
  - Validation: stratified carve-out from `dreaddit_train`
  - Test: fixed `dreaddit_test` (never used for fitting/model selection)
- Cross-domain:
  - Supervised target metrics only if target has >=2 classes.
  - If target unlabeled/1-class, run adaptation-only and skip supervised F1 target claims.

## 4. Data Quality Gates
Configured in `config.yaml` via `data_quality`:
- Duplicate ratio threshold.
- Empty text ratio threshold.
- Required binary labels for Dreaddit datasets.
- Maximum Dreaddit train-test clean-text overlap ratio.

## 5. Known Risks / Limitations
- Counseling has high duplication after cleaning and no valid binary labels for direct supervised evaluation.
- Graph construction is currently transductive unless otherwise stated in experiment notes.

## 6. Reproducibility Artifacts
- EDA outputs: `experiments/results/eda_report.json`, `experiments/results/eda_summary.csv`
- Protocol split artifact: `experiments/results/dreaddit_protocol_splits.json`
- Phase status artifact: `experiments/results/phase_ab_status.json`
