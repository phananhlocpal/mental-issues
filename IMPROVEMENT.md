# IMPROVEMENT PLAN FOR MENTAL-ISSUES EXPERIMENT

## 1. Muc tieu
Tai lieu nay de xuat ke hoach sua toan bo thiet ke thi nghiem de dam bao:
- Dung phuong phap khoa hoc.
- Tranh leakage va over-claim.
- Bao cao ket qua co the tai lap.
- Dat muc chat luong huong den bai bao Q1.

## 2. Van de cot loi da xac nhan
### 2.1 Dataset counseling khong phu hop cho bai toan binary stress detection
- EDA cho thay counseling chi co 1 nhan (label=1).
- Duplicate cao (hon 70% ban ghi trung lap theo clean text).
- Neu tiep tuc dung nhu hien tai, metric cross-domain se bi sai lech.

### 2.2 Protocol danh gia hien tai chua dung chuan
- In-domain dang split ngau nhien tren dreaddit_train thay vi dung dreaddit_test chinh thuc.
- Graph va feature duoc xay dung tu toan bo node truoc khi chia train/val/test (dang transductive), can khai bao ro hoac thay doi protocol.

### 2.3 Mismatch giua proposal va implementation knowledge module
- Proposal huong den knowledge graph/ontology co y nghia hoc thuat.
- Implementation hien tai chu yeu la lexicon rule-based.
- Muc dong gop “knowledge-infused” chua du manh.

### 2.4 Ablation chua day du
- Config khai bao 4 bien the.
- Notebook moi chay 2 bien the.
- Chua du bang chung de ket luan scientific claim.

### 2.5 Metric va bao cao chua day du cho publication
- Dang uu tien binary F1 trong boi canh co domain 1-class.
- Chua bao cao mean+-std nhieu seed.
- Chua co CI, significance test, calibration.

## 3. Ke hoach sua thi nghiem
## Phase A - Data Governance and EDA (bat buoc)
### Muc tieu
Lam sach va xac thuc du lieu truoc khi train.

### Cong viec
1. Tao script EDA chinh thuc xuat report:
- Class distribution tung dataset.
- Duplicate rate truoc/sau cleaning.
- Empty/very-short samples.
- Length distribution.
- Lexical overlap giua cac split/domain.
2. Tao data quality gate:
- Neu 1 split chi co 1 class -> fail pipeline.
- Neu duplicate vuot nguong (vi du 20%) -> canh bao/fail.
- Neu overlap train-test vuot nguong -> fail.
3. Chuan hoa policy xu ly counseling:
- Option 1: Loai counseling khoi bai toan binary neu khong co negative class.
- Option 2: Tai tao nhan hop le (co quy trinh giai thich duoc) neu co co so khoa hoc.
4. Chot dataset card noi bo:
- Nguon, filtering, split protocol, known bias, limitation.

### Tieu chi hoan thanh
- Co file report EDA luu trong experiments/results.
- Pipeline se dung lai neu vi pham data gate.

## Phase B - Redesign Evaluation Protocol (uu tien cao nhat)
### Muc tieu
Dam bao ket qua danh gia hop le va tai lap duoc.

### Cong viec
1. In-domain protocol (khuyen nghi):
- Train: dreaddit_train.
- Validation: carve out tu dreaddit_train (stratified).
- Test: dreaddit_test giu nguyen, khong dung trong bat ky buoc fit nao.
2. Cross-domain protocol:
- Chi chay neu source va target deu co >=2 class.
- Tach ro adaptation unlabeled va target hold-out test.
- Khong dung target test cho model selection.
3. Tach 2 che do danh gia:
- Inductive mode: graph/feature cho test duoc tao tu train-only info.
- Transductive mode: neu dung full graph thi phai ghi ro va benchmark cong bang.
4. Khoa random seed va ghi lai tat ca artifact de reproducibility.

### Tieu chi hoan thanh
- Co mot file protocol.md mo ta dung setup danh gia.
- Co ket qua in-domain hop le tren dreaddit_test.

## Phase C - Improve Data Processing and Feature Pipeline
### Muc tieu
Giam noise, tang tin hieu hoc duoc.

### Cong viec
1. Nang cap preprocessing:
- Batching/token cleaning on train statistics only.
- Bo sung rule xu ly repeated chars, html artifacts, malformed unicode.
2. Duplicate handling:
- De-dup exact va near-duplicate cho train.
- Bao toan unique sample trong test.
3. Label noise audit:
- Kiem tra cac mau outlier va confidence thap.
4. Feature ablation tai muc preprocessing:
- Co/khong stopword removal.
- Co/khong lemmatization.
- max_length, tfidf_top_k, co_occurrence_window.

### Tieu chi hoan thanh
- Co bang so sanh tac dong preprocessing (delta F1, recall class 1).

## Phase D - Model and Knowledge Module Alignment with Proposal
### Muc tieu
Lam cho implementation sat voi y tuong de tai.

### Cong viec
1. Knowledge module:
- Nang cap tu pure keyword sang ontology-aware mapping (neu co UMLS/tuong duong).
- Them confidence score cho entity-concept mapping.
- Bo sung edge typing theo semantic relation (khong chi same-category).
2. Graph module:
- Thiet ke trong so edge ro rang va co chuan hoa.
- Kiem soat sparsity va noise cua edge.
3. Model module:
- Chuan hoa architecture card: input, hidden, heads, loss terms.
- Thu nghiem focal loss/weighted loss neu imbalance.
4. Domain adaptation:
- Chay alpha schedule va lambda sweep co he thong.
- Bao cao khi nao domain adversarial that su giup.

### Tieu chi hoan thanh
- Co minh chung domain adaptation + knowledge module tao gain ben vung qua nhieu seed.

## Phase E - Full Ablation, Baseline, and Statistical Reporting
### Muc tieu
Dat muc bang chung khoa hoc phu hop nop journal.

### Cong viec
1. Chay day du ablation:
- full_model
- no_knowledge_graph
- no_domain_adversarial
- no_symptom_nodes
2. Baseline cong bang:
- TF-IDF + LR/SVM
- Transformer-only baseline
- Homo-GCN/GAT
3. Bao cao metric day du:
- Accuracy, Macro-F1, per-class F1, AUROC, AUPRC.
- Mean+-std tren >=5 seeds.
- Bootstrap CI hoac significance test.
4. Error analysis:
- FP/FN taxonomy.
- Domain-specific failure patterns.

### Tieu chi hoan thanh
- Co bang ket qua tong hop va appendix loi dien hinh.

## 4. Ke hoach sua code theo file
1. src/preprocessing/data_loader.py
- Them data gate cho class cardinality, duplicate, empty text.
- Chuan hoa cach map text/label columns va validate schema.
2. src/graph/graph_builder.py
- Tach che do inductive/transductive.
- Bo sung edge filtering va relation weighting.
3. src/entity_extraction/extractor.py
- Bo sung confidence va mapping policy de giam false mapping.
4. src/training/trainer.py
- Mo rong metric (macro/per-class), logging train protocol.
- Bo sung multi-seed runner.
5. src/evaluation/metrics.py
- Them CI, significance, calibration plots.
6. experiment.ipynb
- Chuyen sang protocol moi: train/val/test chuan.
- Chay day du ablation theo config.
7. config.yaml
- Dong bo bien the ablation va hyperparameter sweep.

## 5. Milestone de xuat (2-3 tuan)
1. Tuan 1:
- Hoan tat Phase A + B.
- Co in-domain benchmark hop le.
2. Tuan 2:
- Hoan tat Phase C + D.
- Co ket qua so bo cho knowledge/domain modules.
3. Tuan 3:
- Hoan tat Phase E.
- Dong bang ket qua, viet section ket qua va han che.

## 6. Definition of Done (publication-ready internal)
1. Khong con leakage da biet trong protocol.
2. Dataset card + EDA report day du.
3. Tat ca ablation khai bao da duoc chay.
4. Bao cao mean+-std nhieu seed + CI/significance.
5. Toan bo ket qua co the tai lap bang 1 command setup + 1 command run.

## 7. Hanh dong ngay lap tuc (Next 48h)
1. Dong bang protocol in-domain dung dreaddit_test.
2. Tam dung cross-domain voi counseling cho den khi co nhan 2 lop hop le.
3. Them data quality gate trong pipeline.
4. Chay lai full baseline + proposed tren protocol moi.
5. Cap nhat performance_table va viet note giai thich thay doi.

## 8. Phase C Progress Update (2026-03-13)
1. Da them preprocessing nang cao trong pipeline:
- normalize unicode
- html unescape + html tag cleanup
- collapse repeated chars
- tuy chon stopword removal / lemmatization
2. Da them de-dup train-only trong `src/preprocessing/data_loader.py`:
- exact duplicate: ON
- near-duplicate: co san logic + threshold, tam thoi OFF theo ket qua benchmark nhanh
3. Da co artifact bao cao preprocessing impact:
- `experiments/results/phase_c_preprocessing_impact.csv`
- `experiments/results/phase_c_preprocessing_top3.csv`
4. Khuyen nghi config hien tai (tam thoi):
- `train_dedup_exact: true`
- `train_dedup_near: false`
- `remove_stopwords: true`
- `use_lemmatization: true`

## 9. Phase D Progress Update (2026-03-13)
1. Knowledge extractor:
- Them `confidence: float` vao `Entity` dataclass (1.0 = exact, 0.85 = multi-token)
- Them `CONCEPT_RELATIONS: dict[tuple, RelationType]` voi 25 typed pairs (comorbid/implies/triggers/co_occurs)
- Export tu `src/entity_extraction/__init__.py`
2. Graph module:
- Per-doc max normalization doc-word TF-IDF weights (`normalize_doc_word_weights: true`)
- Word-word co-occ filtered by `min_cooc_count: 2` va count-normalized
- Word-concept filtered by `min_entity_confidence: 0.8`
- Concept-concept dung CONCEPT_RELATIONS typed edges (weight=1.0) + same-category fallback (0.5)
3. Model:
- Them `FocalLoss(gamma, weight, reduction)` vao `src/models/hgnn.py`
- Export tu `src/models/__init__.py`
4. Trainer:
- Switch focal/CE dua theo `training.use_focal_loss` config
- Log `domain_lambda` va `use_focal_loss` vao moi history row de sweep analysis
5. Config:
- `graph.min_cooc_count: 2`, `normalize_doc_word_weights: true`, `min_entity_confidence: 0.8`
- `training.use_focal_loss: false`, `training.focal_gamma: 2.0`

## 10. Phase E Progress Update (2026-03-13)
1. Metrics nang cao trong `src/evaluation/metrics.py`:
- `compute_metrics()`: bo sung f1_macro, per-class F1 (f1_class0/1), AUROC, AUPRC; compat aliases (f1/precision/recall)
- `bootstrap_ci()`: 1000-sample bootstrap CI voi coverage tuy chinh
- `aggregate_seed_results()`: mean±std tren nhieu seed
- `format_mean_std_table()`: bang ket qua dang publication (mean±std per cell)
- `plot_pr_curve()`: Precision-Recall curve voi AUPRC
- `plot_calibration_curve()`: reliability diagram
- `plot_multi_roc()`: overlay ROC curves nhieu model
- `error_analysis()`: FP/FN taxonomy voi word-level top_words va mean confidence
- `save_error_analysis()`: luu cac bucket thanh CSV
- Export day du trong `src/evaluation/__init__.py`
2. GNNTrainer.evaluate() nang cap:
- Thu thap y_prob tu softmax, tra ve full metrics (AUROC, AUPRC, f1_macro, per-class F1)
3. EmbeddingTrainer.evaluate() nang cap (tuong tu GNNTrainer):
- Thu thap y_prob, dung `compute_metrics()`
4. Runner Phase E:
- `run_phase_e.py`: multi-seed ablation (5 seeds) x 4 variants + 5 baselines
- Tu dong luu: ablation_seed_results.json, baseline_seed_results.json
- Tao: phase_e_mean_std_table.csv + .md, full_model_ci_summary.csv
- Error analysis CSVs cho full_model best seed
- Calibration + PR curves cho full_model
- Tham so: `--seeds N`, `--skip-baselines`, `--skip-ablation`
