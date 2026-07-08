## Cách QA Model hoạt động — Giải thích chi tiết

### 1. Bài toán Question Answering là gì?

Cho một **câu hỏi** và một **đoạn văn bản ngữ cảnh (context)**, mô hình cần **trích xuất ra một đoạn text** trong context làm câu trả lời. Đây gọi là *extractive QA* (không phải generative QA — mô hình không tự sinh ra chữ mới, mà chỉ xác định vị trí bắt đầu và kết thúc của câu trả lời trong context gốc).

Ví dụ:

```
Question: "Which deep learning libraries back 🤗 Transformers?"
Context:  "...🤗 Transformers is backed by the three most popular deep learning libraries
           — Jax, PyTorch, and TensorFlow — with a seamless integration between them..."
Answer:   "Jax, PyTorch, and TensorFlow"
```

Mô hình cần trả lời: câu trả lời bắt đầu ở ký tự thứ 78 và kết thúc ở ký tự 105 trong context.

---

### 2. Input được tổ chức như thế nào?

Tokenized input có cấu trúc:

```
[CLS] question_text [SEP] context_text [SEP]
```

Hình dung cụ thể:

```
Vị trí token: 0      1  2  3  ...  k     k+1  k+2  k+3  ...  n-1   n
Token:       [CLS]  Which deep ...  ?   [SEP]  🤗  Transformers ...  .   [SEP]
                ↑                      ↑      ↑                      ↑
             luôn unmask           question  |______ context ________|
                                  (mask lại)    
```

Mỗi token được gán một **sequence ID**:
- `0` = thuộc câu hỏi
- `1` = thuộc context
- `None` = special token (`[CLS]`, `[SEP]`)

---

### 3. Tại sao model trả về 2 bộ logits thay vì 1?

Không giống như classification (1 logit cho mỗi token), QA model cần dự đoán **2 thứ cùng lúc**: token nào bắt đầu câu trả lời, và token nào kết thúc câu trả lời. Do đó:

- **`start_logits`**: shape `(batch_size, seq_len)` — mỗi vị trí token có 1 giá trị logit thể hiện "token này có khả năng là **điểm bắt đầu** câu trả lời hay không"
- **`end_logits`**: shape `(batch_size, seq_len)` — tương tự, nhưng cho **điểm kết thúc**

**Đây chính là điểm mấu chốt:** Model không dự đoán trực tiếp câu trả lời. Nó dự đoán **vị trí (index)** của token bắt đầu và token kết thúc trong chuỗi input đã tokenized.

```
Input tokens:       [CLS]  Which  deep  ...  Jax  ,  PyTorch  and  TensorFlow  ...  [SEP]
Token index:          0      1      2   ...   21  22    23     24      25      ...   65

start_logits[21] = 8.2  ← cao → token "Jax" có khả năng là START
end_logits[25]   = 7.9  ← cao → token "TensorFlow" có khả năng là END

→ Answer span: tokens 21 đến 25 → "Jax, PyTorch and TensorFlow"
```

---

### 4. Từng bước xử lý chi tiết

#### Bước 1: Tokenize và chạy qua model

```python
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

model_checkpoint = "distilbert-base-cased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

question = "Which deep learning libraries back 🤗 Transformers?"
context = """...🤗 Transformers is backed by the three most popular deep learning libraries
— Jax, PyTorch, and TensorFlow — with a seamless integration between them..."""

inputs = tokenizer(question, context, return_tensors="pt")
# inputs["input_ids"].shape → torch.Size([1, 66])   (66 tokens)

outputs = model(**inputs)
start_logits = outputs.start_logits  # shape: (1, 66)
end_logits   = outputs.end_logits    # shape: (1, 66)
```

66 tokens bao gồm: `[CLS]` + question tokens + `[SEP]` + context tokens + `[SEP]`.

#### Bước 2: Mask các vị trí không được phép dự đoán

Ta **không muốn** model dự đoán start/end nằm trong phần câu hỏi, vì câu trả lời luôn nằm trong context. Ta cũng không muốn dự đoán vào token `[SEP]`.

```
sequence_ids: [None, 0, 0, 0, ..., 0, None, 1, 1, 1, ..., 1, None]
                ↑                   ↑           ↑                  ↑
              [CLS]            question cuối  context bắt đầu   [SEP] cuối
```

→ Mask tất cả vị trí có `sequence_ids != 1` (không phải context), trừ `[CLS]` (vị trí 0) — vì một số model dùng `[CLS]` để biểu thị "không có câu trả lời trong context".

```python
import torch

sequence_ids = inputs.sequence_ids()

# Mask mọi thứ không phải context
mask = [i != 1 for i in sequence_ids]
# Nhưng unmask [CLS] (vị trí 0)
mask[0] = False
mask = torch.tensor(mask)[None]  # shape: (1, 66)

start_logits[mask] = -10000  # Gán giá trị cực âm → sau softmax ≈ 0
end_logits[mask]   = -10000
```

**Tại sao lại là `-10000`?** Vì sau đó ta dùng softmax. `exp(-10000)` ≈ 0, nên những vị trí này sẽ có xác suất gần như bằng 0.

#### Bước 3: Softmax để có xác suất

```python
start_probabilities = torch.nn.functional.softmax(start_logits, dim=-1)[0]  # shape: (66,)
end_probabilities   = torch.nn.functional.softmax(end_logits, dim=-1)[0]    # shape: (66,)
```

Sau bước này, `start_probabilities[i]` là xác suất token thứ `i` là **START** của câu trả lời. Tương tự cho `end_probabilities`.

#### Bước 4: Tìm cặp (start, end) tối ưu — đây là phần quan trọng nhất!

Nếu chỉ đơn giản lấy `argmax(start_probs)` và `argmax(end_probs)`, ta có thể gặp trường hợp **start > end** (vô lý). Do đó cần tính **xác suất đồng thời** của mọi cặp (start, end) hợp lệ.

Với giả định sự kiện "bắt đầu tại i" và "kết thúc tại j" là **độc lập**:

```
P(start=i, end=j) = P(start=i) × P(end=j)
```

Ta tạo ma trận `scores` kích thước `(66, 66)`:

```python
# outer product: mỗi phần tử (i,j) = start_probs[i] × end_probs[j]
scores = start_probabilities[:, None] * end_probabilities[None, :]
# shape: (66, 66)
```

Minh họa ma trận `scores`:

```
              end=0   end=1   end=2   ...  end=65
start=0    [ 0.001   0.002   0.000   ...   0.000 ]
start=1    [ 0.000   0.000   0.000   ...   0.000 ]   ← đã bị mask
start=21   [ 0.001   0.003   0.005   ...   0.082 ]   ← "Jax" là start
start=22   [ ...     ...     ...     ...   ...   ]
...
start=65   [ 0.000   0.000   0.000   ...   0.000 ]
```

Sau đó dùng **`triu` (upper triangular)** để chỉ giữ lại các cặp `start ≤ end`:

```python
scores = torch.triu(scores)
# Các ô start > end → bị gán = 0
```

Minh họa sau `triu`:

```
              end=0   end=1   end=2   ...  end=25  ...  end=65
start=0    [ 0.001   0.002   0.000   ...   0.001  ...   0.000 ]
start=1    [ 0.000   0.000   0.000   ...   0.000  ...   0.000 ]
start=21   [ 0.000   0.000   0.000   ...   0.082  ...   0.015 ]  ← highlight
start=22   [ 0.000   0.000   0.000   ...   0.000  ...   0.000 ]
...
start=26   [ 0.000   0.000   0.000   ...   0.000  ...   0.000 ]  ← 0 vì start > end
```

#### Bước 5: Lấy vị trí có score cao nhất

```python
max_index = scores.argmax().item()  # vị trí trong tensor đã flatten

# Chuyển từ index phẳng → tọa độ (row, col)
start_index = max_index // scores.shape[1]  # 21
end_index   = max_index % scores.shape[1]   # 25

score = scores[start_index, end_index].item()  # 0.97773
```

#### Bước 6: Convert token indices → vị trí ký tự trong context gốc

Đây là lúc **offset mapping** phát huy tác dụng:

```python
inputs_with_offsets = tokenizer(question, context, return_offsets_mapping=True)
offsets = inputs_with_offsets["offset_mapping"]
# offsets: [(0,0), (0,5), (6,10), ..., (start_char, end_char), ...]

start_char, _ = offsets[start_index]  # vị trí ký tự bắt đầu trong context
_, end_char   = offsets[end_index]    # vị trí ký tự kết thúc

answer = context[start_char:end_char]  # "Jax, PyTorch and TensorFlow"
```

Tổng kết:

```python
result = {
    "answer": answer,
    "start": start_char,   # 78
    "end": end_char,       # 105
    "score": score,        # 0.97773
}
```

---

### 5. Tóm tắt trực quan toàn bộ pipeline

```
┌──────────────┐    ┌──────────────┐    ┌─────────────────────┐
│  Tokenizer   │ →  │    Model     │ →  │  Post-processing    │
│              │    │              │    │                     │
│ question +   │    │ start_logits │    │ 1. Mask non-context │
│ context      │    │ end_logits   │    │ 2. Softmax          │
│              │    │ (1, seq_len) │    │ 3. Outer product    │
│              │    │              │    │ 4. triu (start≤end) │
│ input_ids    │    │              │    │ 5. argmax           │
│ (1, 66)      │    │              │    │ 6. Offset → chars   │
└──────────────┘    └──────────────┘    └─────────────────────┘
                                                 ↓
                     {"answer": "Jax, PyTorch and TensorFlow",
                      "start": 78, "end": 105, "score": 0.97773}
```

---

### 6. Điểm mấu chốt cần nhớ

| Khái niệm | Giải thích |
|---|---|
| **start_logits** | Mỗi token có 1 giá trị: token đó có phải là **vị trí bắt đầu** của answer không |
| **end_logits** | Mỗi token có 1 giá trị: token đó có phải là **vị trí kết thúc** của answer không |
| **Masking** | Câu trả lời chỉ nằm trong context → mask hết question + `[SEP]`; giữ `[CLS]` cho trường hợp "không có câu trả lời" |
| **Softmax riêng biệt** | `start_probs` và `end_probs` được softmax độc lập (không phải chung 1 softmax) |
| **Outer product** | `scores[i][j] = start_probs[i] × end_probs[j]` — tính xác suất đồng thời |
| **triu** | Chỉ giữ `start ≤ end` — loại bỏ cặp vô nghĩa |
| **argmax** | Tìm cặp (i, j) có score cao nhất |
| **Offset mapping** | Chuyển từ token index → vị trí ký tự thực trong text gốc |

Model được train để làm tốt việc này: trong quá trình training, loss function sẽ phạt nếu start_logits tại vị trí đúng thấp, và tương tự cho end_logits. Hai đầu ra này được học **độc lập** nhưng cùng chia sẻ phần thân Transformer phía trước.