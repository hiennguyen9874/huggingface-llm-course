# 🚀 Chương 1: Giới thiệu về Transformer và Large Language Models (LLMs)

## 1. Tổng quan khóa học

Khóa học này dạy về **Large Language Models (LLMs)** và **Xử lý ngôn ngữ tự nhiên (NLP)** sử dụng hệ sinh thái Hugging Face:

- 🤗 **Transformers** – thư viện cốt lõi để làm việc với các mô hình Transformer
- 🤗 **Datasets** – thư viện xử lý dữ liệu
- 🤗 **Tokenizers** – thư viện tokenization
- 🤗 **Accelerate** – thư viện giúp huấn luyện phân tán dễ dàng
- **Hugging Face Hub** – kho chứa hàng triệu mô hình pretrained

**Yêu cầu:** Biết Python tốt, đã học qua deep learning cơ bản. Không yêu cầu kiến thức PyTorch/TensorFlow trước.

---

## 2. NLP và LLMs là gì?

### NLP (Natural Language Processing)
Là lĩnh vực kết hợp ngôn ngữ học và machine learning, tập trung vào việc giúp máy tính **hiểu ngữ cảnh** của ngôn ngữ con người (không chỉ từng từ riêng lẻ).

**Các tác vụ NLP phổ biến:**

| Loại tác vụ | Ví dụ |
|---|---|
| **Phân loại cả câu** | Phân tích cảm xúc (tích cực/tiêu cực), phát hiện spam, kiểm tra ngữ pháp |
| **Phân loại từng từ** | Xác định từ loại (danh từ, động từ), nhận dạng thực thể (người, địa điểm) |
| **Sinh văn bản** | Hoàn thành câu, điền từ vào chỗ trống |
| **Trích xuất câu trả lời** | Trả lời câu hỏi dựa trên ngữ cảnh cho sẵn |
| **Sinh câu mới từ đầu vào** | Dịch máy, tóm tắt văn bản |

NLP không chỉ giới hạn ở văn bản – nó còn xử lý **giọng nói** (nhận dạng giọng nói) và **thị giác máy tính** (mô tả ảnh).

### LLMs (Large Language Models)
Là một **tập con mạnh mẽ** của NLP, được huấn luyện trên lượng dữ liệu khổng lồ:

- **Quy mô**: Hàng triệu đến hàng trăm tỷ tham số
- **Khả năng tổng quát**: Thực hiện nhiều tác vụ mà không cần huấn luyện riêng
- **In-context learning**: Học từ ví dụ được cung cấp ngay trong prompt
- **Emergent abilities**: Khả năng xuất hiện bất ngờ khi mô hình đủ lớn

**Hạn chế của LLMs:**
- **Ảo giác (Hallucination)**: Tự tin tạo ra thông tin sai
- **Thiếu hiểu biết thực sự**: Chỉ hoạt động dựa trên mẫu thống kê
- **Thiên kiến (Bias)**: Tái tạo định kiến từ dữ liệu huấn luyện
- **Cửa sổ ngữ cảnh hữu hạn**: Giới hạn số token có thể xử lý
- **Tiêu tốn tài nguyên**: Cần GPU mạnh để chạy

---

## 3. Transformers có thể làm gì? – Hàm `pipeline()`

Đối tượng cơ bản nhất trong 🤗 Transformers là `pipeline()`. Nó kết nối mô hình với các bước tiền xử lý và hậu xử lý, cho phép nhập văn bản thô và nhận kết quả ngay:

```python
from transformers import pipeline

# 1. Phân tích cảm xúc
classifier = pipeline("sentiment-analysis")
result = classifier("I've been waiting for a HuggingFace course my whole life.")
# [{'label': 'POSITIVE', 'score': 0.9598}]

# Nhiều câu cùng lúc
results = classifier([
    "I love this!",
    "I hate this!"
])
```

**3 bước bên trong pipeline:**
1. Văn bản được **tiền xử lý** thành định dạng mô hình hiểu được
2. Đầu vào đã xử lý được đưa qua **mô hình**
3. Dự đoán được **hậu xử lý** thành kết quả dễ đọc

### Các pipeline quan trọng:

#### a) Zero-shot Classification – Phân loại không cần huấn luyện
```python
classifier = pipeline("zero-shot-classification")
result = classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
)
# {'labels': ['education', 'business', 'politics'],
#  'scores': [0.8446, 0.1120, 0.0434]}
```
Cực kỳ mạnh mẽ: không cần fine-tune, chỉ cần đưa danh sách nhãn mong muốn.

#### b) Text Generation – Sinh văn bản
```python
generator = pipeline("text-generation")
result = generator("In this course, we will teach you how to", 
                   max_length=30, 
                   num_return_sequences=2)
```
Có thể **chọn mô hình cụ thể** từ Hub:
```python
generator = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-360M")
```

#### c) Fill-Mask – Điền vào chỗ trống
```python
unmasker = pipeline("fill-mask")
result = unmasker("This course will teach you all about <mask> models.", top_k=2)
# [{'sequence': '...mathematical models.', 'score': 0.196},
#  {'sequence': '...computational models.', 'score': 0.040}]
```
> ⚠️ Token mask khác nhau tùy mô hình: BERT dùng `[MASK]`, một số khác dùng `<mask>`.

#### d) Named Entity Recognition (NER) – Nhận dạng thực thể
```python
ner = pipeline("ner", aggregation_strategy="simple")
result = ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")
# PER: Sylvain, ORG: Hugging Face, LOC: Brooklyn
```
`aggregation_strategy="simple"` giúp gộp các từ thuộc cùng một thực thể (ví dụ: "Hugging" + "Face").

#### e) Question Answering – Trả lời câu hỏi
```python
qa = pipeline("question-answering")
result = qa(
    question="Where do I work?",
    context="My name is Sylvain and I work at Hugging Face in Brooklyn",
)
# {'answer': 'Hugging Face', 'score': 0.638}
```

#### f) Summarization – Tóm tắt
```python
summarizer = pipeline("summarization")
result = summarizer(long_text, max_length=50, min_length=20)
```

#### g) Translation – Dịch máy
```python
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
result = translator("Ce cours est produit par Hugging Face.")
# {'translation_text': 'This course is produced by Hugging Face.'}
```

#### h) Xử lý ảnh và âm thanh
```python
# Phân loại ảnh
image_classifier = pipeline("image-classification", model="google/vit-base-patch16-224")

# Nhận dạng giọng nói
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3")
```

---

## 4. Kiến trúc Transformer – Cách chúng hoạt động

### Lịch sử ngắn gọn
| Thời gian | Mô hình | Đặc điểm |
|---|---|---|
| 06/2017 | **Transformer** (bản gốc) | Kiến trúc nền tảng, thiết kế cho dịch máy |
| 06/2018 | **GPT** | Pretrained Transformer đầu tiên, fine-tune cho nhiều tác vụ |
| 10/2018 | **BERT** | Hiểu ngữ cảnh hai chiều (bidirectional) |
| 02/2019 | **GPT-2** | Lớn hơn, mạnh hơn GPT |
| 10/2019 | **T5** | Sequence-to-sequence đa nhiệm |
| 05/2020 | **GPT-3** | Zero-shot learning, không cần fine-tune |
| 01/2023 | **Llama** (Meta) | Mã nguồn mở, đa ngôn ngữ |
| 03/2023 | **Mistral** | 7B tham số, vượt Llama 2 13B |
| 05/2024 | **Gemma 2** (Google) | Nhẹ, hiệu quả, 2B–27B tham số |
| 11/2024 | **SmolLM2** | 135M–1.7B tham số, chạy trên thiết bị di động |

### Language Models – Mô hình ngôn ngữ
Tất cả Transformer đều được huấn luyện như **language models** theo kiểu **self-supervised** (tự tạo nhãn từ dữ liệu, không cần con người gán nhãn):

- **Causal Language Modeling (CLM):** Dự đoán từ tiếp theo trong câu (chỉ nhìn từ phía trước). Dùng cho GPT, Llama.
- **Masked Language Modeling (MLM):** Dự đoán từ bị che trong câu (nhìn cả hai phía). Dùng cho BERT.

### Transfer Learning – Học chuyển giao
Đây là ý tưởng cốt lõi: **Pretraining** (huấn luyện từ đầu trên dữ liệu khổng lồ) → **Fine-tuning** (tinh chỉnh trên dữ liệu chuyên biệt nhỏ hơn).

**Lợi ích của fine-tuning:**
- Tận dụng kiến thức đã học từ pretraining
- Cần ít dữ liệu hơn
- Tốn ít thời gian và tài nguyên hơn
- Kết quả tốt hơn huấn luyện từ đầu

### Cấu trúc chung của Transformer

Mô hình Transformer gồm hai khối chính:

```
┌──────────────────────────────────────────────┐
│                  TRANSFORMER                  │
│                                               │
│   ┌──────────┐          ┌──────────┐         │
│   │ ENCODER  │          │ DECODER  │         │
│   │          │          │          │         │
│   │ (Hiểu)   │ ───────→ │ (Sinh)   │         │
│   │          │ features  │          │         │
│   └──────────┘          └──────────┘         │
└──────────────────────────────────────────────┘
```

- **Encoder (bên trái):** Nhận input, xây dựng biểu diễn (features). Tối ưu cho việc **hiểu**.
- **Decoder (bên phải):** Dùng features từ encoder + các input khác để **sinh** output. Tối ưu cho việc **tạo ra**.

### Attention – Cơ chế chú ý (trái tim của Transformer)

> **"Attention Is All You Need"** – Tiêu đề bài báo gốc năm 2017

Attention cho phép mô hình **tập trung vào những từ quan trọng** trong câu khi xử lý từng từ.

**Ví dụ:** Khi dịch "You like this course" sang tiếng Pháp:
- Để dịch đúng từ "like" → cần chú ý đến "You" (chia động từ theo chủ ngữ)
- Để dịch đúng từ "this" → cần chú ý đến "course" (giống đực/cái trong tiếng Pháp)

### Ba biến thể kiến trúc chính

| Kiến trúc | Cơ chế Attention | Đại diện | Phù hợp cho |
|---|---|---|---|
| **Encoder-only** | Bidirectional (2 chiều) | BERT, DistilBERT, ModernBERT | Phân loại câu, NER, trả lời câu hỏi trích xuất |
| **Decoder-only** | Unidirectional (1 chiều, trái→phải) | GPT, Llama, Gemma, SmolLM | Sinh văn bản, hội thoại, sáng tạo |
| **Encoder-Decoder** | Encoder: 2 chiều / Decoder: 1 chiều | BART, T5, Marian | Dịch máy, tóm tắt, sinh câu trả lời |

---

## 5. Các mô hình giải quyết tác vụ cụ thể như thế nào?

### A. Sinh văn bản (GPT-2 – Decoder-only)

```
 Prompt → [Token Embedding + Positional Encoding] 
        → Nhiều Decoder Block (masked self-attention)
        → Language Modeling Head 
        → Dự đoán token tiếp theo
```

GPT-2 dùng **masked self-attention**: chỉ được nhìn các token phía trước (bên trái), không được nhìn tương lai. Huấn luyện theo kiểu **Causal Language Modeling**.

### B. Phân loại văn bản (BERT – Encoder-only)

```
Input → [Token Embedding + Segment Embedding + Positional Encoding]
      → Nhiều Encoder Block (bidirectional attention)
      → Lấy hidden state của token [CLS]
      → Classification Head → Logits → Nhãn
```

BERT dùng **WordPiece tokenization**, thêm token `[CLS]` ở đầu (dùng cho classification) và `[SEP]` để phân cách câu. Huấn luyện với 2 mục tiêu: **Masked Language Modeling** và **Next Sentence Prediction**.

### C. Tóm tắt/Dịch máy (BART/T5 – Encoder-Decoder)

```
Input → Encoder (BERT-like, bidirectional) → Features
                                              ↓
Output ← Decoder (GPT-like, autoregressive) ←┘
```

BART huấn luyện bằng cách **làm hỏng input** (text infilling: thay nhiều đoạn văn bản bằng một token `[MASK]`) rồi bắt decoder **tái tạo lại** văn bản gốc.

### D. Xử lý giọng nói (Whisper – Encoder-Decoder)
- **Encoder:** Chuyển audio → log-Mel spectrogram → Transformer encoder
- **Decoder:** Tự sinh transcript token-by-token
- Huấn luyện trên 680,000 giờ audio đa ngôn ngữ

### E. Thị giác máy tính (ViT – Vision Transformer)
- Chia ảnh thành các **patch** (ví dụ: ảnh 224×224 → 196 patch 16×16)
- Mỗi patch → vector embedding
- Thêm `[CLS]` token + position embedding
- Cho qua Transformer encoder
- `[CLS]` output → classification head

---

## 6. Suy luận (Inference) với LLMs

### Hai pha chính trong inference:

#### Pha 1: Prefill (Tiền xử lý)
1. **Tokenization**: Chuyển văn bản → tokens
2. **Embedding**: Token → vector số
3. **Xử lý khởi tạo**: Qua toàn bộ mạng neural để hiểu ngữ cảnh

→ Tốn nhiều tính toán, nhưng chỉ làm **một lần** cho input.

#### Pha 2: Decode (Sinh văn bản)
Lặp lại cho mỗi token mới:
1. **Attention computation**: Nhìn lại tất cả token trước đó
2. **Probability calculation**: Tính xác suất cho từng token tiếp theo
3. **Token selection**: Chọn token
4. **Continuation check**: Tiếp tục hay dừng?

→ Tốn nhiều bộ nhớ, sinh từng token một (autoregressive).

### Các chiến lược chọn token (Sampling Strategies):

```
Raw Logits (xác suất thô của tất cả token)
    ↓
Temperature (điều chỉnh độ "sáng tạo"):
  - >1.0: sáng tạo, đa dạng hơn
  - <1.0: tập trung, deterministic hơn
    ↓
Top-k: chỉ giữ k token có xác suất cao nhất
Top-p (Nucleus): chỉ giữ các token có tổng xác suất đạt ngưỡng p
    ↓
Chọn token cuối cùng
```

### Kiểm soát lặp từ:
- **Presence penalty**: Phạt cố định nếu token đã xuất hiện
- **Frequency penalty**: Phạt tỉ lệ với số lần xuất hiện

### Beam Search:
Thay vì chọn 1 token mỗi bước, duy trì **nhiều chuỗi ứng viên** song song (thường 5-10). Giúp output mạch lạc hơn nhưng tốn tài nguyên hơn.

### Các chỉ số hiệu năng:
| Chỉ số | Ý nghĩa |
|---|---|
| **TTFT** (Time to First Token) | Thời gian có token đầu tiên – ảnh hưởng UX |
| **TPOT** (Time Per Output Token) | Tốc độ sinh từng token tiếp theo |
| **Throughput** | Số request xử lý đồng thời |
| **VRAM Usage** | Bộ nhớ GPU cần thiết |

### KV Cache:
Kỹ thuật tối ưu quan trọng: **lưu lại** Key-Value từ các bước trước đó để không phải tính lại, giúp tăng tốc đáng kể (đánh đổi bằng bộ nhớ).

---

## 7. Thiên kiến và Giới hạn (Bias & Limitations)

> ⚠️ **Cảnh báo quan trọng khi dùng mô hình trong production!**

Mô hình được huấn luyện trên dữ liệu scrape từ internet – bao gồm cả nội dung độc hại, phân biệt chủng tộc, giới tính.

**Ví dụ thực tế với BERT:**

```python
from transformers import pipeline

unmasker = pipeline("fill-mask", model="bert-base-uncased")

# Với "man":
print(unmasker("This man works as a [MASK]."))
# ['lawyer', 'carpenter', 'doctor', 'waiter', 'mechanic']

# Với "woman":
print(unmasker("This woman works as a [MASK]."))
# ['nurse', 'waitress', 'teacher', 'maid', 'prostitute']
```

BERT – dù được huấn luyện trên dữ liệu "trung lập" (Wikipedia + BookCorpus) – vẫn thể hiện **định kiến giới tính rõ rệt**.

**Bài học:** Fine-tune trên dữ liệu của bạn **không loại bỏ được** thiên kiến có sẵn từ pretraining. Luôn kiểm tra và đánh giá output của mô hình trước khi triển khai.

---

## 8. Cách chọn kiến trúc phù hợp

| Tác vụ | Kiến trúc đề xuất | Mô hình ví dụ |
|---|---|---|
| Phân loại văn bản (cảm xúc, chủ đề) | **Encoder** | BERT, RoBERTa |
| Sinh văn bản sáng tạo | **Decoder** | GPT, Llama |
| Dịch máy | **Encoder-Decoder** | T5, BART |
| Tóm tắt | **Encoder-Decoder** | BART, T5 |
| Nhận dạng thực thể (NER) | **Encoder** | BERT, RoBERTa |
| Trả lời câu hỏi (trích xuất) | **Encoder** | BERT, RoBERTa |
| Trả lời câu hỏi (sinh) | **Encoder-Decoder** hoặc **Decoder** | T5, GPT |
| Hội thoại / Chatbot | **Decoder** | GPT, Llama |

**Nguyên tắc chọn nhanh:**
1. Cần **hiểu** ngữ cảnh 2 chiều? → Encoder
2. Cần **sinh** văn bản mới? → Decoder
3. Cần **chuyển đổi** một chuỗi thành chuỗi khác? → Encoder-Decoder

---

## 9. Tổng kết – Những điểm cốt lõi

1. **`pipeline()`** là cách đơn giản nhất để sử dụng mô hình pretrained cho nhiều tác vụ
2. **Attention** là trái tim của Transformer – cho phép mô hình tập trung vào từ quan trọng trong ngữ cảnh
3. **Transfer Learning** (Pretrain → Fine-tune) tiết kiệm tài nguyên và cho kết quả tốt hơn
4. **3 kiến trúc chính:** Encoder-only (hiểu), Decoder-only (sinh), Encoder-Decoder (chuyển đổi)
5. **LLMs** mạnh mẽ nhưng có giới hạn: ảo giác, thiên kiến, tốn tài nguyên
6. **Inference** gồm 2 pha: Prefill (xử lý input) và Decode (sinh từng token)
7. **Sampling strategies** (temperature, top-k, top-p) kiểm soát chất lượng và độ sáng tạo của output
8. Luôn **kiểm tra thiên kiến** trước khi triển khai mô hình

---

Trên đây là toàn bộ nội dung chi tiết của Chương 1. Các chương tiếp theo sẽ đi sâu vào cách sử dụng thư viện Transformers để load, fine-tune và triển khai mô hình.