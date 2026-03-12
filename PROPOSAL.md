Dưới đây là **đề cương nghiên cứu hoàn chỉnh** tổng hợp toàn bộ các nội dung đã thảo luận, được viết theo cấu trúc thường dùng trong **proposal / research proposal cho AI & Healthcare**. Nội dung tập trung vào: bối cảnh, xu hướng nghiên cứu, tổng hợp phương pháp, research gap và đề xuất hướng nghiên cứu.

---

# 1. Bối cảnh và tầm quan trọng của vấn đề nghiên cứu

Rối loạn sức khỏe tâm thần đang trở thành một trong những thách thức lớn nhất đối với y tế toàn cầu. Theo báo cáo của World Health Organization, hàng trăm triệu người trên thế giới đang phải đối mặt với các vấn đề như trầm cảm, lo âu và stress kéo dài. Tuy nhiên, khả năng tiếp cận dịch vụ chẩn đoán và hỗ trợ tâm lý vẫn còn hạn chế, đặc biệt tại các quốc gia đang phát triển.

Sự phát triển của **trí tuệ nhân tạo và xử lý ngôn ngữ tự nhiên (NLP)** mở ra cơ hội lớn trong việc tự động phát hiện các dấu hiệu tâm lý từ dữ liệu văn bản. Các nguồn dữ liệu như bài đăng trên mạng xã hội, nhật ký cá nhân, hoặc hội thoại tư vấn tâm lý có thể phản ánh các biểu hiện tâm lý quan trọng. Những nghiên cứu gần đây cho thấy việc phân tích các tín hiệu ngôn ngữ có thể giúp phát hiện sớm các trạng thái tâm lý tiêu cực như stress hoặc trầm cảm.

Tuy nhiên, các hệ thống hiện tại vẫn gặp nhiều hạn chế. Phần lớn các phương pháp chỉ dựa vào **mô hình ngôn ngữ học thống kê hoặc transformer** như BERT và RoBERTa. Những mô hình này có khả năng học biểu diễn ngữ nghĩa mạnh mẽ nhưng thường **không tích hợp tri thức y khoa có cấu trúc**, dẫn đến việc thiếu khả năng suy luận lâm sàng.

Ngoài ra, nhiều nghiên cứu chỉ được huấn luyện trên một loại dữ liệu duy nhất (ví dụ: mạng xã hội), dẫn đến hiệu suất suy giảm khi áp dụng sang các miền dữ liệu khác như hội thoại lâm sàng. Bên cạnh đó, khả năng **giải thích kết quả (explainability)** của các mô hình hiện tại vẫn còn hạn chế, trong khi đây là yêu cầu quan trọng đối với các hệ thống AI trong y tế.

Do đó, việc phát triển các phương pháp kết hợp **tri thức y khoa, mô hình đồ thị và cơ chế giải thích** đang trở thành hướng nghiên cứu quan trọng nhằm nâng cao độ chính xác và độ tin cậy của các hệ thống đánh giá sức khỏe tâm thần tự động.

---

# 2. Xu hướng nghiên cứu chính trong lĩnh vực

Trong những năm gần đây, nghiên cứu về phát hiện rối loạn tâm lý từ dữ liệu văn bản đã phát triển mạnh mẽ với ba xu hướng chính.

## 2.1 Transformer-based mental health detection

Nhiều nghiên cứu sử dụng các mô hình ngôn ngữ tiền huấn luyện như:

* BERT
* RoBERTa
* ClinicalBERT

Các mô hình này có khả năng trích xuất biểu diễn ngữ nghĩa mạnh mẽ từ văn bản và đạt kết quả tốt trong nhiều bài toán phân loại cảm xúc và phát hiện stress.

Tuy nhiên, chúng thường xử lý văn bản theo **chuỗi tuyến tính** và không khai thác cấu trúc tri thức bên ngoài.

---

## 2.2 Graph Neural Networks cho NLP

Để khắc phục hạn chế của các mô hình chuỗi, nhiều nghiên cứu bắt đầu sử dụng **mạng nơ-ron đồ thị (Graph Neural Networks)** để mô hình hóa mối quan hệ giữa các thực thể trong văn bản.

Một số mô hình phổ biến bao gồm:

* Graph Convolutional Network
* Graph Attention Network
* Heterogeneous Graph Attention Network

Những phương pháp này cho phép mô hình hóa các quan hệ như:

* quan hệ đồng xuất hiện giữa các từ
* quan hệ giữa tài liệu và từ vựng
* quan hệ giữa các khái niệm

Nhờ đó, mô hình có thể thực hiện **suy luận đa bước (multi-hop reasoning)**.

---

## 2.3 Knowledge-enhanced NLP

Một xu hướng nổi bật khác là tích hợp **knowledge graph** vào các hệ thống NLP.

Các ontology y sinh học lớn như:

* UMLS

cung cấp hàng triệu khái niệm và quan hệ y khoa, bao gồm:

```text
symptom_of
associated_with
is_a
```

Việc tích hợp tri thức này giúp mô hình hiểu mối liên hệ giữa các triệu chứng và rối loạn tâm lý, từ đó cải thiện khả năng suy luận.

---

# 3. Các phương pháp tiếp cận đã được sử dụng

Các phương pháp hiện tại có thể chia thành ba nhóm chính.

## 3.1 Phương pháp dựa trên đặc trưng văn bản

Các nghiên cứu ban đầu sử dụng:

* Bag-of-Words
* TF-IDF
* SVM hoặc logistic regression

Các phương pháp này đơn giản nhưng không nắm bắt được ngữ cảnh sâu.

---

## 3.2 Phương pháp deep learning

Các mô hình deep learning như CNN, LSTM và transformer đã được áp dụng rộng rãi. Trong đó, các mô hình transformer như BERT cho phép học biểu diễn ngữ nghĩa mạnh mẽ và đạt hiệu suất cao trên nhiều dataset.

---

## 3.3 Phương pháp dựa trên đồ thị

Các nghiên cứu gần đây áp dụng GNN để xây dựng đồ thị giữa các từ, tài liệu hoặc khái niệm.

Ví dụ:

* Graph-based text classification
* Knowledge graph reasoning
* Heterogeneous graph modeling

Các phương pháp này có khả năng khai thác cấu trúc quan hệ phức tạp giữa các thực thể.

---

# 4. Tổng hợp kết quả từ các công trình tiêu biểu

Các nghiên cứu gần đây cho thấy:

1. Transformer-based models đạt hiệu suất cao trong phát hiện stress và trầm cảm từ văn bản.
2. Graph neural networks giúp cải thiện khả năng mô hình hóa quan hệ giữa các từ và khái niệm.
3. Knowledge graph giúp tăng khả năng suy luận và giảm phụ thuộc vào dữ liệu huấn luyện lớn.
4. Các phương pháp explainable AI giúp tăng tính minh bạch của mô hình.

Tuy nhiên, phần lớn các nghiên cứu vẫn còn hạn chế ở một số khía cạnh quan trọng.

---

# 5. Khoảng trống nghiên cứu (Research Gaps)

Từ việc tổng hợp các công trình hiện tại, có thể xác định một số khoảng trống nghiên cứu chính:

### Gap 1 — Thiếu tích hợp tri thức y khoa

Phần lớn các mô hình chỉ dựa vào dữ liệu văn bản và không sử dụng tri thức từ các ontology y khoa.

---

### Gap 2 — Thiếu khả năng tổng quát hóa giữa các miền dữ liệu

Các mô hình thường được huấn luyện trên một dataset duy nhất và hoạt động kém khi áp dụng sang dữ liệu khác.

Ví dụ:

* dữ liệu mạng xã hội
* hội thoại tư vấn tâm lý

---

### Gap 3 — Khả năng giải thích còn hạn chế

Các phương pháp phổ biến như:

* SHAP
* LIME

chỉ cung cấp mức giải thích dựa trên đặc trưng, chưa thể hiện được chuỗi suy luận.

---

# 6. Hướng nghiên cứu đề xuất

Để giải quyết các khoảng trống trên, nghiên cứu này đề xuất một phương pháp mới dựa trên **Knowledge-Infused Heterogeneous Graph Neural Networks**.

Phương pháp đề xuất bao gồm các thành phần chính:

### 1. Trích xuất thực thể y khoa từ văn bản

Sử dụng các mô hình NLP để nhận diện các triệu chứng tâm lý trong văn bản.

---

### 2. Xây dựng đồ thị không đồng nhất

Đồ thị bao gồm các loại node:

```
Document node
Word node
Medical concept node
Symptom category node
```

Các quan hệ trong đồ thị bao gồm:

```
document-word
word-word
word-concept
concept-category
concept-concept
```

---

### 3. Huấn luyện mô hình graph neural network

Áp dụng:

* Heterogeneous Graph Attention Network

để học biểu diễn của các node trong đồ thị.

---

### 4. Học biểu diễn đa miền

Sử dụng kỹ thuật:

* Domain Adversarial Training
* Gradient Reversal Layer

nhằm học các đặc trưng tâm lý không phụ thuộc vào domain.

---

### 5. Cơ chế giải thích

Áp dụng:

* GNNExplainer

để trích xuất các **subgraph giải thích**, thể hiện chuỗi suy luận từ biểu hiện ngôn ngữ đến các khái niệm y khoa.

---

# 7. Dataset sử dụng

Nghiên cứu dự kiến sử dụng hai dataset thuộc hai miền dữ liệu khác nhau:

* Dreaddit
* Mental Health Counseling Conversations

Việc sử dụng hai dataset giúp đánh giá khả năng **cross-domain generalization**.

---

# 8. Các bài báo học thuật quan trọng (WoS/Scopus)

Một số công trình quan trọng liên quan trực tiếp đến đề tài bao gồm:

1. Yao et al. (2019) — Graph Convolutional Networks for Text Classification.
2. Velickovic et al. (2018) — Graph Attention Networks.
3. Wang et al. (2019) — Heterogeneous Graph Attention Network.
4. Devlin et al. (2019) — BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.
5. Ying et al. (2019) — GNNExplainer: Generating Explanations for Graph Neural Networks.
6. Turcan & McKeown (2019) — Dreaddit: A Reddit Dataset for Stress Analysis.
7. Ganin et al. (2016) — Domain-Adversarial Training of Neural Networks.

Các công trình này đều được lập chỉ mục trong **Web of Science hoặc Scopus** và đóng vai trò nền tảng cho nghiên cứu đề xuất.

---

# 9. Kết luận

Nghiên cứu đề xuất một phương pháp mới kết hợp **knowledge graph, heterogeneous graph neural networks, domain adaptation và explainable AI** nhằm nâng cao hiệu quả phát hiện các dấu hiệu tâm lý từ dữ liệu văn bản.

Phương pháp này không chỉ cải thiện độ chính xác mà còn cung cấp **khả năng giải thích mang ý nghĩa lâm sàng**, giúp tăng tính tin cậy của các hệ thống AI trong lĩnh vực chăm sóc sức khỏe tâm thần.

---
