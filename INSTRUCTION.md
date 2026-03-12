# 1. Mục tiêu thí nghiệm

Mục tiêu của thí nghiệm là xây dựng và đánh giá một hệ thống **phát hiện trạng thái tâm lý từ dữ liệu văn bản** bằng cách kết hợp:

* mô hình ngôn ngữ
* đồ thị tri thức y khoa
* mạng nơ-ron đồ thị
* cơ chế học đa miền
* phương pháp giải thích mô hình

Cụ thể, hệ thống cần:

1. Phân loại văn bản thành các trạng thái tâm lý (ví dụ: stress / non-stress).
2. Học được các biểu diễn chung giữa **nhiều miền dữ liệu khác nhau**.
3. Cung cấp **giải thích dựa trên cấu trúc đồ thị** cho dự đoán của mô hình.

---

# 2. Tổng quan pipeline hệ thống

Pipeline tổng thể gồm các bước:

```
Text Data
↓
Preprocessing
↓
Entity Extraction
↓
Knowledge Mapping
↓
Heterogeneous Graph Construction
↓
Graph Neural Network Training
↓
Domain Adversarial Learning
↓
Prediction
↓
Explainability Analysis
```

Các thành phần quan trọng gồm:

* embedding từ mô hình ngôn ngữ
* đồ thị không đồng nhất
* mô hình GNN
* module domain adaptation
* module giải thích

---

# 3. Môi trường triển khai

### 3.1 Phần cứng

Yêu cầu tối thiểu:

```
GPU: ≥12GB VRAM
RAM: ≥16GB
Storage: ≥50GB
```

GPU đề xuất:

```
RTX 3060 / RTX 3080 / RTX 4090
```

---

### 3.2 Phần mềm

Hệ điều hành:

```
Ubuntu 20.04+ hoặc Windows 11
```

Phiên bản Python:

```
Python 3.10
```

Thư viện chính:

```
PyTorch
PyTorch Geometric
Transformers
datasets
scikit-learn
pandas
numpy
matplotlib
```

---

# 4. Dataset

Hai dataset sẽ được sử dụng để đánh giá **cross-domain learning**.

### Dataset 1

Dreaddit

Đặc điểm:

```
Nguồn: Reddit
Nhãn: stress / non-stress
Ngôn ngữ: informal
```

---

### Dataset 2

Mental Health Counseling Conversations

Đặc điểm:

```
Nguồn: hội thoại tư vấn tâm lý
Ngôn ngữ: bán lâm sàng
```

---

### Cách tải dataset

Sử dụng thư viện Hugging Face:

```python
from datasets import load_dataset

dreaddit = load_dataset("dreaddit")
```

---

# 5. Tiền xử lý dữ liệu

Các bước preprocessing cần thực hiện:

### 5.1 Text cleaning

Thực hiện:

```
lowercase
remove URLs
remove emojis
remove punctuation
```

---

### 5.2 Tokenization

Sử dụng tokenizer từ:

* BERT

Ví dụ:

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
```

---

### 5.3 Stopword removal

Loại bỏ các từ không mang ý nghĩa:

```
the
is
and
```

---

### 5.4 Lemmatization

Chuẩn hóa từ về dạng gốc.

---

# 6. Trích xuất thực thể y khoa

Mục tiêu là nhận diện các **triệu chứng tâm lý** trong văn bản.

Các khái niệm được tham chiếu từ:

* UMLS

---

### Quy trình

1. NER detection
2. Entity normalization
3. Concept mapping

---

### Ví dụ

Input text:

```
I can't sleep and feel exhausted all the time.
```

Entities:

```
sleep problem → insomnia
exhausted → fatigue
```

---

# 7. Xây dựng đồ thị không đồng nhất

Đồ thị sẽ bao gồm **4 loại node**.

### Node types

```
Document
Word
Medical Concept
Symptom Category
```

---

### Edge types

Các quan hệ:

```
Document → Word
Word → Word
Word → Medical Concept
Medical Concept → Symptom Category
Medical Concept → Medical Concept
```

---

### Edge weight

Document–Word:

```
TF-IDF
```

Word–Word:

```
co-occurrence window = 5
```

---

### Kích thước đồ thị dự kiến

Nếu dataset có:

```
10k documents
```

Graph sẽ có khoảng:

```
30k – 50k nodes
200k+ edges
```

---

# 8. Embedding của node

### Document node

Embedding từ:

* SciBERT

Kích thước vector:

```
768
```

---

### Word node

Embedding:

```
GloVe
```

---

### Medical concept node

Embedding:

```
BERT embedding từ definition
```

---

### Symptom category node

Embedding:

```
mean embedding của các concept
```

---

# 9. Mô hình Graph Neural Network

Mô hình chính sử dụng:

* Heterogeneous Graph Attention Network

---

### Kiến trúc

```
Input node features
↓
Heterogeneous Graph Attention Layer
↓
Graph Embedding
↓
Classifier
```

---

### Hyperparameters

```
hidden_dim = 128
num_heads = 8
dropout = 0.5
learning_rate = 0.001
batch_size = 32
epochs = 50
```

---

# 10. Domain adversarial training

Áp dụng kỹ thuật:

* Domain Adversarial Training

Sử dụng:

* Gradient Reversal Layer

---

### Mục tiêu

Học representation:

```
psychological signals
```

không phụ thuộc:

```
Reddit
Counseling data
```

---

### Loss function

```
Total Loss = Classification Loss + λ * Domain Loss
```

---

# 11. Baseline models

Các mô hình so sánh:

### Text models

* BERT
* RoBERTa

---

### Graph models

* Graph Convolutional Network
* Graph Attention Network

---

# 12. Thiết kế thí nghiệm

### Experiment 1 — In-domain evaluation

```
Train: Dreaddit
Test: Dreaddit
```

---

### Experiment 2 — Cross-domain evaluation

```
Train: Dreaddit
Test: Counseling dataset
```

và

```
Train: Counseling
Test: Dreaddit
```

---

### Experiment 3 — Ablation study

Các biến thể cần thử:

```
Full model
Without knowledge graph
Without domain adversarial
Without symptom nodes
```

---

# 13. Evaluation metrics

Các metric sử dụng:

```
Accuracy
Precision
Recall
F1-score
```

F1-score là metric chính.

---

# 14. Explainability

Sử dụng:

* GNNExplainer

Mục tiêu:

Trích xuất **subgraph giải thích** cho từng dự đoán.

---

### Ví dụ output

```
Text
↓
hopeless
↓
depression
↓
mood disorder
```

---

# 15. Phân tích lỗi

Sau khi huấn luyện, cần phân tích:

```
false positives
false negatives
```

Các trường hợp thường gây lỗi:

```
sarcasm
implicit emotion
slang
```

---

# 16. Kết quả cần báo cáo

Báo cáo cuối cùng phải bao gồm:

```
performance table
ablation table
cross-domain results
training cost
```

Ngoài ra cần:

```
embedding visualization
explanation graphs
```

---

# 17. Lịch trình thực hiện

| Giai đoạn           | Thời gian |
| ------------------- | --------- |
| Dataset preparation | 1 tuần    |
| Preprocessing       | 1 tuần    |
| Graph construction  | 2 tuần    |
| Model training      | 2 tuần    |
| Experiments         | 2 tuần    |
| Analysis            | 1 tuần    |

Tổng thời gian:

```
8–9 tuần
```

---

# 18. Deliverables

Trợ lý nghiên cứu cần cung cấp:

```
Source code
Processed dataset
Graph data
Experiment logs
Performance tables
Visualization figures
```

---

✅ **Kết quả mong đợi**

Hệ thống đề xuất dự kiến:

```
outperform baseline models
generalize across domains
provide interpretable reasoning
```

để phục vụ cho **công bố khoa học trong các tạp chí AI/Healthcare thuộc WoS hoặc Scopus**.
