# Tổng hợp Chương 2: Sử Dụng Thư Viện 🤗 Transformers

---

## 1. Giới Thiệu Về 🤗 Transformers

Các mô hình Transformer thường rất lớn (hàng triệu đến hàng chục tỉ tham số), việc huấn luyện và triển khai rất phức tạp. Thư viện 🤗 Transformers ra đời để cung cấp một API thống nhất cho phép load, train và lưu bất kỳ mô hình Transformer nào.

**Ba đặc điểm cốt lõi của thư viện:**

| Đặc điểm | Mô tả |
|----------|-------|
| **Dễ sử dụng** | Chỉ với 2 dòng code đã có thể load và inference một mô hình NLP state-of-the-art |
| **Linh hoạt** | Mọi model đều là `nn.Module` của PyTorch, có thể xử lý như bất kỳ model ML nào |
| **Đơn giản** | "All in one file" — toàn bộ forward pass của một model nằm trong một file duy nhất, dễ đọc và dễ hack |

Khác biệt lớn nhất với các thư viện ML khác: mỗi model có layer riêng, không dùng chung module giữa các file. Điều này cho phép thử nghiệm trên một model mà không ảnh hưởng đến model khác.

---

## 2. Đằng Sau Hàm `pipeline()`

Hàm `pipeline()` gộp 3 bước: **preprocessing → model → postprocessing**.

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier([
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
])
# Output: [{'label': 'POSITIVE', 'score': 0.9598}, {'label': 'NEGATIVE', 'score': 0.9995}]
```

### 2.1 Preprocessing với Tokenizer

Transformer không xử lý text thô được → cần tokenizer chuyển text thành số.

**Tokenizer thực hiện 3 việc:**
1. Tách input thành các token (từ, subword, ký hiệu)
2. Map mỗi token sang một số nguyên (ID)
3. Thêm các input phụ trợ cần thiết

```python
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
```

Kết quả là dictionary chứa:
- **`input_ids`**: ma trận 2 hàng (mỗi hàng là 1 câu), mỗi phần tử là ID của một token
- **`attention_mask`**: ma trận cùng shape, 1 = token cần chú ý, 0 = padding token bị bỏ qua

### 2.2 Đưa Input Qua Model

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(checkpoint)
outputs = model(**inputs)
print(outputs.last_hidden_state.shape)  # torch.Size([2, 16, 768])
```

**Vector đầu ra của Transformer có 3 chiều:**
- **Batch size** (2): số câu xử lý cùng lúc
- **Sequence length** (16): độ dài biểu diễn số của chuỗi
- **Hidden size** (768): chiều vector của mỗi input — đây là lý do gọi là "high-dimensional" (có thể lên tới 3072+ với model lớn)

> **Quan trọng:** Output của 🤗 Transformers hoạt động như `namedtuple` hoặc dictionary. Có thể truy cập qua attribute (`outputs.last_hidden_state`), key (`outputs["last_hidden_state"]`), hoặc index (`outputs[0]`).

### 2.3 Model Head: Biến Hidden States Thành Dự Đoán

Hidden states từ Transformer base được đưa vào **head** — thường là một hoặc vài linear layer — để chuyển sang không gian phù hợp với task.

Các loại head phổ biến (tiền tố `*` là tên model):
- `*Model` — lấy hidden states
- `*ForSequenceClassification` — phân loại chuỗi
- `*ForCausalLM` — sinh văn bản (language modeling)
- `*ForMaskedLM` — masked language modeling
- `*ForQuestionAnswering` — hỏi đáp
- `*ForTokenClassification` — phân loại token (NER)

```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
outputs = model(**inputs)
print(outputs.logits.shape)  # torch.Size([2, 2]) — 2 câu, 2 nhãn
```

### 2.4 Postprocessing: Từ Logits Đến Xác Suất

Logits là raw scores chưa chuẩn hóa. Để thành xác suất phải qua **Softmax**:

```python
import torch

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)
# tensor([[0.0402, 0.9598],   # câu 1: NEGATIVE 4%, POSITIVE 96%
#         [0.9995, 0.0005]])  # câu 2: NEGATIVE 99.95%, POSITIVE 0.05%

model.config.id2label  # {0: 'NEGATIVE', 1: 'POSITIVE'}
```

> **Lưu ý:** 🤗 Transformers luôn trả về **logits** (không phải xác suất), vì loss function thường kết hợp Softmax với cross-entropy.

---

## 3. Models — Chi Tiết Kỹ Thuật

### 3.1 Tạo Model

```python
from transformers import AutoModel

# AutoModel tự đoán kiến trúc phù hợp từ checkpoint
model = AutoModel.from_pretrained("bert-base-cased")

# Hoặc chỉ định trực tiếp class
from transformers import BertModel
model = BertModel.from_pretrained("bert-base-cased")
```

`from_pretrained()` tải và cache model từ Hugging Face Hub.

### 3.2 Lưu và Load Model

```python
model.save_pretrained("directory_on_my_computer")
# Tạo 2 file: config.json + model.safetensors

# Load lại
model = AutoModel.from_pretrained("directory_on_my_computer")
```

- **`config.json`**: chứa metadata, version, và các thuộc tính kiến trúc model
- **`model.safetensors`**: state dict chứa tất cả weights của model

Đẩy model lên Hub:
```python
from huggingface_hub import notebook_login
notebook_login()

model.push_to_hub("my-awesome-model")
# Người khác có thể load: AutoModel.from_pretrained("your-username/my-awesome-model")
```

### 3.3 Mã Hóa Văn Bản (Encoding)

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
encoded_input = tokenizer("Hello, I'm a single sentence!")
print(encoded_input)
# {'input_ids': [101, 8667, 117, ...], 
#  'token_type_ids': [0, 0, 0, ...], 
#  'attention_mask': [1, 1, 1, ...]}

tokenizer.decode(encoded_input["input_ids"])
# '[CLS] Hello, I'\m a single sentence! [SEP]'
```

**Ba trường trong output:**

| Trường | Ý nghĩa |
|--------|---------|
| `input_ids` | Biểu diễn số của các token |
| `token_type_ids` | Phân biệt câu A và câu B trong pair tasks (sẽ học ở Chương 3) |
| `attention_mask` | Token nào được attention (1) và token nào bị bỏ qua (0) |

### 3.4 Padding — Làm Cho Các Chuỗi Cùng Độ Dài

```python
encoded_input = tokenizer(
    ["How are you?", "I'm fine, thank you!"], 
    padding=True, 
    return_tensors="pt"
)
# Câu ngắn hơn được thêm padding token (ID = 0)
# attention_mask tương ứng = 0 để model bỏ qua
```

### 3.5 Truncation — Cắt Chuỗi Quá Dài

BERT chỉ xử lý được tối đa 512 token. Nếu chuỗi dài hơn, dùng `truncation=True`:

```python
encoded_input = tokenizer(
    "This is a very very ... long sentence.",
    truncation=True,  # Cắt về max_length của model
)

# Kết hợp padding + truncation với max_length tùy chỉnh
encoded_input = tokenizer(
    ["How are you?", "I'm fine, thank you!"],
    padding=True, truncation=True, max_length=5, return_tensors="pt"
)
```

### 3.6 Special Tokens

BERT và các model tương tự dùng token đặc biệt:
- `[CLS]` (ID: 101): đánh dấu đầu câu
- `[SEP]` (ID: 102): đánh dấu cuối câu / phân cách 2 câu

Tokenizer **tự động thêm** các token này vì model được pretrain với chúng.

---

## 4. Tokenizers — Phân Loại & Cơ Chế

Tokenizer chuyển text → số. Có 3 loại chính:

### 4.1 Word-based Tokenization

Tách text thành từ (bằng dấu cách, dấu câu). 

**Nhược điểm:**
- Vocabulary cực lớn (tiếng Anh >500k từ)
- "dog" và "dogs" không liên quan gì với nhau
- Nhiều unknown tokens `[UNK]` nếu từ chưa có trong vocab

```python
"Jim Henson was a puppeteer".split()
# ['Jim', 'Henson', 'was', 'a', 'puppeteer']
```

### 4.2 Character-based Tokenization

Tách text thành từng ký tự.

**Ưu điểm:** Vocab nhỏ, gần như không có unknown tokens.

**Nhược điểm:**
- Mỗi ký tự mang ít ý nghĩa
- Chuỗi đầu vào rất dài (một từ → 10+ tokens) → tốn tài nguyên tính toán

### 4.3 Subword Tokenization ⭐ (Quan Trọng Nhất)

**Nguyên tắc:** Từ phổ biến giữ nguyên, từ hiếm tách thành subword có nghĩa.

Ví dụ: "tokenization" → "token" + "ization"

**Ưu điểm kết hợp:**
- Vocab nhỏ gọn, ít unknown tokens
- Subword mang ý nghĩa ngữ nghĩa
- Hiệu quả với ngôn ngữ chắp dính (Turkish, Finnish...)

**Các thuật toán phổ biến:**

| Thuật toán | Sử dụng trong |
|------------|---------------|
| Byte-level BPE | GPT-2 |
| WordPiece | BERT |
| SentencePiece / Unigram | Các model đa ngôn ngữ |

### 4.4 Encoding & Decoding

**Encoding (text → IDs)** gồm 2 bước:
```python
# Bước 1: Tokenization
tokens = tokenizer.tokenize("Using a Transformer network is simple")
# ['Using', 'a', 'transform', '##er', 'network', 'is', 'simple']
# '##' nghĩa là token này nối liền với token trước

# Bước 2: Convert tokens → IDs
ids = tokenizer.convert_tokens_to_ids(tokens)
# [7993, 170, 11303, 1200, 2443, 1110, 3014]
```

**Decoding (IDs → text):**
```python
tokenizer.decode([7993, 170, 11303, 1200, 2443, 1110, 3014])
# 'Using a Transformer network is simple'
# Tokenizer tự gộp các subword để tạo text có thể đọc được
```

---

## 5. Xử Lý Nhiều Chuỗi Cùng Lúc

### 5.1 Model Luôn Yêu Cầu Batch

```python
# ❌ Sai — thiếu batch dimension
input_ids = torch.tensor(ids)  # shape: (seq_len,)
model(input_ids)  # IndexError!

# ✅ Đúng — thêm batch dimension
input_ids = torch.tensor([ids])  # shape: (1, seq_len)
model(input_ids)  # OK!
```

### 5.2 Padding Để Tạo Tensor Hình Chữ Nhật

Khi batch các câu có độ dài khác nhau, cần **padding** để có tensor chữ nhật. Token padding được thêm vào cuối câu ngắn:

```
input_ids:    [[200, 200, 200],       →  [[200, 200, 200],
               [200, 200]]               [200, 200,   0]]
                                     padding ID = 0
```

### 5.3 Attention Mask — Bắt Buộc Phải Có Khi Padding

**Vấn đề:** Attention layer trong Transformer contextualize **tất cả** token, bao gồm cả padding. Nếu không dùng mask, padding làm sai lệch kết quả.

```python
batched_ids = [
    [200, 200, 200],
    [200, 200, tokenizer.pad_token_id],
]

attention_mask = [
    [1, 1, 1],
    [1, 1, 0],  # 0 = bỏ qua padding
]

outputs = model(torch.tensor(batched_ids), 
                attention_mask=torch.tensor(attention_mask))
```

> **Kết quả:** Logits của batch = logits khi xử lý riêng từng câu. Nếu không dùng mask, logits sẽ bị sai.

### 5.4 Chuỗi Quá Dài

Hầu hết model giới hạn ở 512-1024 token. Hai giải pháp:
1. Dùng model hỗ trợ chuỗi dài: **Longformer**, **LED**
2. **Truncate** — cắt ngắn chuỗi: `sequence = sequence[:max_sequence_length]`

---

## 6. API Tokenizer Cấp Cao — Gộp Tất Cả

Thay vì gọi riêng `tokenize()`, `convert_tokens_to_ids()`, padding thủ công, **gọi trực tiếp tokenizer** xử lý tất cả:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

sequences = ["I've been waiting for a HuggingFace course my whole life.", 
             "So have I!"]

# Một dòng xử lý tất cả
tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
output = model(**tokens)
```

**Các chế độ padding:**
```python
tokenizer(sequences, padding="longest")   # pad đến câu dài nhất
tokenizer(sequences, padding="max_length")  # pad đến max của model (512)
tokenizer(sequences, padding="max_length", max_length=8)  # pad đến 8
```

**Các chế độ truncation:**
```python
tokenizer(sequences, truncation=True)  # cắt về max của model
tokenizer(sequences, max_length=8, truncation=True)  # cắt về 8
```

**Loại tensor trả về:**
```python
tokenizer(sequences, return_tensors="pt")  # PyTorch tensors
tokenizer(sequences, return_tensors="np")  # NumPy arrays
```

---

## 7. Triển Khai Inference Tối Ưu Trong Production

Chương này cũng đề cập 3 framework để deploy LLM trong môi trường production:

### 7.1 So Sánh TGI, vLLM, llama.cpp

| Tiêu chí | TGI | vLLM | llama.cpp |
|----------|-----|------|-----------|
| **Ngôn ngữ** | Rust + Python | Python | C/C++ |
| **Quản lý bộ nhớ** | Flash Attention 2 + Continuous Batching | PagedAttention | Quantization + CPU optimization |
| **Triển khai** | Docker + K8s + Monitoring | Ray clusters | Lightweight server |
| **Phù hợp** | Enterprise production | High-performance serving | Consumer hardware / edge |

### 7.2 Cơ Chế Bộ Nhớ

**Flash Attention (TGI):** Tối ưu attention bằng cách giảm thiểu data transfer giữa HBM và SRAM trên GPU. Load data một lần vào SRAM và tính toán tại chỗ.

**PagedAttention (vLLM):** Chia KV cache thành các "page" nhỏ như virtual memory của OS. Cho phép lưu không liên tục, chia sẻ bộ nhớ, tăng throughput lên đến 24x.

**Quantization (llama.cpp):** Giảm precision từ FP32/FP16 → INT8/INT4 để chạy model tỉ tham số trên consumer hardware.

### 7.3 Điều Khiển Sinh Văn Bản

Các tham số quan trọng:
- **Temperature** (0.1-2.0): cao = sáng tạo hơn, thấp = deterministic
- **Top-p (Nucleus sampling)**: chỉ sample từ tập token có tổng xác suất >= p
- **Top-k**: chỉ sample từ k token có xác suất cao nhất
- **Repetition penalty**: phạt token đã xuất hiện để tránh lặp

---

## 8. Tóm Tắt Kiến Thức Cần Nắm Vững

1. **Pipeline = Tokenizer → Model → Postprocessing**
2. **Tokenizer** chuyển text → `input_ids` + `attention_mask`
3. **Model output** có 3 chiều: (batch_size, sequence_length, hidden_size)
4. **Model head** biến hidden states thành task-specific output (logits)
5. **Logits** cần qua Softmax để thành xác suất
6. **Subword tokenization** là chuẩn hiện đại (BPE, WordPiece, Unigram)
7. **Padding + Attention mask** là bắt buộc khi batch các câu khác độ dài
8. **Tokenizer và model phải cùng checkpoint** — nếu không output sẽ vô nghĩa
9. **AutoModel/AutoTokenizer** tự động chọn kiến trúc đúng từ checkpoint
10. `tokenizer()` gọi trực tiếp (thay vì từng bước riêng lẻ) — đây là API chính