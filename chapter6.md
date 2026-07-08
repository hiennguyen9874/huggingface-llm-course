# Chương 6: Tokenizer trong Hugging Face — Tổng hợp chi tiết

---

## 1. Giới thiệu

Khi fine-tune một mô hình (như trong Chapter 3), ta dùng lại tokenizer đã được pretrain cùng mô hình. Nhưng nếu muốn **train mô hình từ đầu (from scratch)** trên một miền dữ liệu hoặc ngôn ngữ hoàn toàn mới, việc dùng tokenizer cũ sẽ kém hiệu quả. Ví dụ: tokenizer train trên tiếng Anh sẽ hoạt động tệ với tiếng Nhật vì cách dùng khoảng trắng và dấu câu khác biệt.

Chương này dạy cách **train một tokenizer mới hoàn toàn** trên corpus của bạn, sử dụng thư viện **🤗 Tokenizers** (viết bằng Rust, cung cấp các "fast tokenizer" trong Transformers).

Các chủ đề chính:
- Train tokenizer mới dựa trên tokenizer có sẵn
- Sức mạnh đặc biệt của fast tokenizer
- Ba thuật toán subword tokenization: BPE, WordPiece, Unigram
- Xây dựng tokenizer từ scratch với thư viện Tokenizers

---

## 2. Train tokenizer mới từ tokenizer cũ (`train_new_from_iterator`)

### Khác biệt giữa "train tokenizer" và "train model"

| | Train model | Train tokenizer |
|---|---|---|
| Phương pháp | Stochastic Gradient Descent (ngẫu nhiên) | Thống kê (deterministic) |
| Mục tiêu | Giảm loss qua từng batch | Xác định subword nào tốt nhất cho corpus |
| Kết quả | Khác nhau mỗi lần nếu không set seed | Luôn giống nhau với cùng thuật toán và corpus |

### Thu thập corpus

Dùng `datasets` để tải dữ liệu. Ví dụ với CodeSearchNet (Python code):

```python
from datasets import load_dataset

raw_datasets = load_dataset("code_search_net", "python")
```

### Tạo iterator để tránh load toàn bộ vào RAM

**Quan trọng:** Dùng generator thay vì list để tiết kiệm bộ nhớ:

```python
def get_training_corpus():
    return (
        raw_datasets["train"][i : i + 1000]["whole_func_string"]
        for i in range(0, len(raw_datasets["train"]), 1000)
    )

training_corpus = get_training_corpus()
```

Generator chỉ load 1000 texts mỗi lần cần. Lưu ý: generator object chỉ dùng được 1 lần, nên wrap trong hàm để tạo generator mới khi cần.

### Train tokenizer mới

```python
from transformers import AutoTokenizer

old_tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)
```

- `train_new_from_iterator()` chỉ hoạt động với **fast tokenizer**
- Tokenizer mới giữ nguyên thuật toán (BPE cho GPT-2), chỉ thay đổi **vocabulary** dựa trên corpus mới

Kết quả: Tokenizer mới học được các token đặc thù cho Python code như `ĊĠĠĠ` (indentation), `Ġ"""` (docstring), và tách đúng camelCase (`LinearLayer` → `["ĠLinear", "Layer"]`).

### Lưu và chia sẻ tokenizer

```python
tokenizer.save_pretrained("code-search-net-tokenizer")
tokenizer.push_to_hub("code-search-net-tokenizer")
```

---

## 3. Sức mạnh đặc biệt của Fast Tokenizer

### Slow vs Fast tokenizer

| | Fast tokenizer | Slow tokenizer |
|---|---|---|
| Ngôn ngữ | Rust (🤗 Tokenizers) | Python thuần |
| Tốc độ batch | **10.8s** | 4ph41s |
| Tính năng | Offset mapping, word_ids, ... | Cơ bản |

### `BatchEncoding` object

Đầu ra của tokenizer là object `BatchEncoding` (subclass của dict), cung cấp thêm các phương thức hữu ích:

```python
encoding = tokenizer("My name is Sylvain and I work at Hugging Face in Brooklyn.")
```

#### Các phương thức quan trọng:

**`tokens()`** — Lấy danh sách token:
```python
encoding.tokens()
# ['[CLS]', 'My', 'name', 'is', 'S', '##yl', '##va', '##in', ...]
```

**`word_ids()`** — Mỗi token thuộc từ nào (index của từ gốc):
```python
encoding.word_ids()
# [None, 0, 1, 2, 3, 3, 3, 3, ...]
```
- `None` = special token (`[CLS]`, `[SEP]`)
- Token index 3,4,5,6 đều map về từ "Sylvain" → cực kỳ hữu ích cho NER, whole word masking

**`word_to_chars(n)`** — Lấy vị trí ký tự bắt đầu/kết thúc của từ thứ n:
```python
start, end = encoding.word_to_chars(3)
example[start:end]  # 'Sylvain'
```

**`sentence_ids()`** — Mỗi token thuộc câu nào (0 hoặc 1 cho cặp câu).

**`char_to_token(n)` / `token_to_chars(n)`** — Map giữa vị trí ký tự và token.

#### Offset Mapping

Khi gọi tokenizer với `return_offsets_mapping=True`:

```python
inputs_with_offsets = tokenizer(example, return_offsets_mapping=True)
inputs_with_offsets["offset_mapping"]
# [(0,0), (0,2), (3,7), (8,10), (11,12), (12,14), ...]
```

Mỗi tuple `(start, end)` cho biết token đó đến từ khoảng ký tự nào trong text gốc. `(0,0)` dành cho special token.

### Ứng dụng: Tái tạo pipeline `token-classification` (NER)

**1. Lấy dự đoán thô:**

```python
from transformers import AutoTokenizer, AutoModelForTokenClassification

model_checkpoint = "dbmdz/bert-large-cased-finetuned-conll03-english"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForTokenClassification.from_pretrained(model_checkpoint)

inputs = tokenizer(example, return_tensors="pt")
outputs = model(**inputs)

probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
predictions = outputs.logits.argmax(dim=-1)[0]
```

**2. Map về nhãn và vị trí:**

```python
results = []
inputs_with_offsets = tokenizer(example, return_offsets_mapping=True)
offsets = inputs_with_offsets["offset_mapping"]

for idx, pred in enumerate(predictions):
    label = model.config.id2label[pred]
    if label != "O":
        start, end = offsets[idx]
        results.append({
            "entity": label,
            "score": probabilities[idx][pred].item(),
            "word": tokens[idx],
            "start": start,
            "end": end,
        })
```

**3. Gom nhóm entity (grouping entities):**

Nguyên tắc: Gom các token liên tiếp có cùng nhãn `I-XXX` (và token đầu có thể là `B-XXX` hoặc `I-XXX`). Dừng khi gặp `O`, nhãn khác, hoặc `B-XXX` cùng loại (đánh dấu entity mới).

Dùng offset để lấy span text chính xác trong câu gốc:
```python
example[33:45]  # 'Hugging Face'
```

**IOB1 vs IOB2:** Hai format đánh nhãn BIO. IOB2 dùng `B-` cho token đầu entity, IOB1 chỉ dùng `B-` để phân tách hai entity liền kề cùng loại.

---

## 4. Fast Tokenizer trong QA Pipeline

### Cách QA model hoạt động

QA model trả về **hai bộ logits**: `start_logits` và `end_logits` — dự đoán vị trí bắt đầu và kết thúc của câu trả lời trong context.

```python
start_logits = outputs.start_logits  # shape: (batch, seq_len)
end_logits = outputs.end_logits      # shape: (batch, seq_len)
```

### Masking token không thuộc context

Input format: `[CLS] question [SEP] context [SEP]`

Dùng `sequence_ids()` để xác định:
- `0` = token câu hỏi
- `1` = token context
- `None` = special token

Mask tất cả token không phải context (trừ `[CLS]`) bằng `-10000` trước softmax.

### Tìm (start, end) tốt nhất

Tính `scores = start_probabilities[:, None] * end_probabilities[None, :]`, sau đó dùng `triu` để chỉ giữ `start <= end`:

```python
scores = start_probs[:, None] * end_probs[None, :]
scores = torch.triu(scores)
max_index = scores.argmax().item()
start_index = max_index // scores.shape[1]
end_index = max_index % scores.shape[1]
```

### Xử lý context dài (truncation + stride)

Khi context quá dài (vượt `max_length`, mặc định 384), QA pipeline **chia context thành các chunk chồng lấp** (stride mặc định 128):

```python
inputs = tokenizer(
    question, long_context,
    stride=128, max_length=384,
    truncation="only_second",    # chỉ truncate context, không truncate question
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
)
```

Kết quả: `overflow_to_sample_mapping` cho biết mỗi chunk thuộc sample nào. Model dự đoán start/end trên từng chunk, sau đó chọn cặp có score cao nhất toàn cục.

---

## 5. Chuẩn hóa (Normalization) và Tiền tokenization (Pre-tokenization)

Pipeline tokenization gồm:
```
Raw text → Normalization → Pre-tokenization → Model → Post-processing → Tokens
```

### Normalization

Làm sạch text: lowercase, bỏ dấu, chuẩn hóa Unicode (NFD, NFKD), xóa khoảng trắng thừa,...

```python
# BERT uncased
tokenizer.backend_tokenizer.normalizer.normalize_str("Héllò hôw are ü?")
# 'hello how are u?'
```

### Pre-tokenization

Chia text thành các "từ" (word-level) trước khi áp dụng model subword:

| Tokenizer | Cách pre-tokenize | Ký hiệu space |
|-----------|-------------------|---------------|
| BERT | Tách trên whitespace + punctuation | (bỏ qua) |
| GPT-2 | Tách trên whitespace + punctuation | `Ġ` |
| T5 (SentencePiece) | Chỉ tách trên whitespace | `▁` |

### SentencePiece

- Coi text là chuỗi Unicode characters, thay space bằng `▁`
- **Reversible tokenization**: decode = concatenate tokens + replace `▁` → space
- Kết hợp với Unigram, **không cần pre-tokenization** → cực hữu ích cho ngôn ngữ không dùng space (Trung, Nhật)

---

## 6. Byte-Pair Encoding (BPE)

**Sử dụng bởi:** GPT, GPT-2, RoBERTa, BART, DeBERTa

### Nguyên lý

BPE **bắt đầu từ vocabulary nhỏ** (các ký tự đơn) và **học các quy tắc merge** dần dần.

**Training algorithm:**

1. Tách tất cả từ thành ký tự
2. Đếm tần suất mọi cặp token liền kề
3. Merge cặp xuất hiện nhiều nhất
4. Lặp lại cho đến khi đạt vocab size mong muốn

Ví dụ với corpus: `("hug",10), ("pug",5), ("pun",12), ("bun",4), ("hugs",5)`

```
Bước 1: ("u","g") → "ug"      (20 lần)
Bước 2: ("u","n") → "un"      (16 lần)
Bước 3: ("h","ug") → "hug"    (15 lần)
...
```

**Tokenization:** Áp dụng tuần tự các merge rules đã học lên từ mới.

### Byte-level BPE

**GPT-2 / RoBERTa** dùng byte-level BPE: thay vì Unicode characters, làm việc ở mức bytes. Base vocab chỉ 256 bytes nhưng có thể biểu diễn mọi ký tự — **không bao giờ gặp unknown token**.

### Triển khai BPE (code minh họa)

```python
def compute_pair_freqs(splits):
    pair_freqs = defaultdict(int)
    for word, freq in word_freqs.items():
        split = splits[word]
        if len(split) == 1: continue
        for i in range(len(split) - 1):
            pair = (split[i], split[i + 1])
            pair_freqs[pair] += freq
    return pair_freqs

def merge_pair(a, b, splits):
    for word in word_freqs:
        split = splits[word]
        i = 0
        while i < len(split) - 1:
            if split[i] == a and split[i + 1] == b:
                split = split[:i] + [a + b] + split[i + 2:]
            else:
                i += 1
        splits[word] = split
    return splits
```

---

## 7. WordPiece

**Sử dụng bởi:** BERT, DistilBERT, MobileBERT, MPNET

### Khác biệt chính với BPE

| | BPE | WordPiece |
|---|---|---|
| Chọn cặp merge | Cặp **xuất hiện nhiều nhất** | Cặp có **score cao nhất** |
| Score formula | (tần suất cặp) | `freq_pair / (freq_first × freq_second)` |
| Encoding | Áp dụng merge rules | Tìm **longest subword** matching từ đầu |

**Ý nghĩa score:** Ưu tiên merge các cặp mà mỗi thành phần riêng lẻ ít phổ biến. Ví dụ: `("un", "##able")` sẽ không được merge ngay vì cả `"un"` và `"##able"` đều rất phổ biến.

### Prefix `##`

WordPiece dùng prefix `##` để đánh dấu token là phần tiếp theo của một từ (không phải token đầu). Ví dụ: `"hugging"` → `["hugg", "##ing"]`.

### Encoding (khác BPE)

Thay vì áp dụng merge rules, WordPiece tìm **subword dài nhất có trong vocab tính từ đầu từ**, rồi lặp lại cho phần còn lại. Nếu không tìm thấy subword nào → toàn bộ từ thành `[UNK]`.

```
"hugs": "hug" có trong vocab → ["hug", "##s"]
```

---

## 8. Unigram

**Sử dụng bởi:** XLNet, ALBERT, T5, mBART, Big Bird (qua SentencePiece)

### Nguyên lý ngược

Unigram **bắt đầu từ vocabulary lớn** (tất cả substrings phổ biến) và **loại bỏ dần token** ít quan trọng.

### Training algorithm

1. Khởi tạo vocab lớn (qua BPE hoặc Enhanced Suffix Array)
2. Với vocab hiện tại, tính **loss** trên toàn corpus (negative log likelihood)
3. Với mỗi token, tính xem loss tăng bao nhiêu nếu xóa token đó
4. Xóa p% token có loss increase thấp nhất (thường p=10-20%)
5. Lặp lại cho đến kích thước mong muốn

### Tokenization algorithm (Viterbi)

Unigram coi mỗi token là độc lập. Xác suất token = tần suất / tổng tần suất.

Tìm segmentation có xác suất cao nhất = **Viterbi algorithm**:

```python
def encode_word(word, model):
    best_segmentations = [{"start": 0, "score": 1}] + \
        [{"start": None, "score": None} for _ in range(len(word))]
    
    for start_idx in range(len(word)):
        best_score_at_start = best_segmentations[start_idx]["score"]
        for end_idx in range(start_idx + 1, len(word) + 1):
            token = word[start_idx:end_idx]
            if token in model and best_score_at_start is not None:
                score = model[token] + best_score_at_start
                if (best_segmentations[end_idx]["score"] is None or
                    best_segmentations[end_idx]["score"] > score):
                    best_segmentations[end_idx] = {"start": start_idx, "score": score}
    
    # Backtrack để lấy tokens
    ...
```

---

## 9. Tổng quan 3 thuật toán

| | BPE | WordPiece | Unigram |
|---|---|---|---|
| **Training** | Vocab nhỏ → merge dần | Vocab nhỏ → merge dần | Vocab lớn → prune dần |
| **Tiêu chí merge** | Cặp xuất hiện nhiều nhất | Score = freq_pair/(freq_a×freq_b) | Xóa token ít ảnh hưởng loss nhất |
| **Kết quả học** | Merge rules + vocab | Chỉ vocab | Vocab + score mỗi token |
| **Encoding** | Áp dụng merge rules | Longest subword matching | Viterbi (tối ưu xác suất) |

---

## 10. Xây dựng tokenizer từ scratch với 🤗 Tokenizers

Kiến trúc thư viện gồm các block:
- `normalizers` — Làm sạch text
- `pre_tokenizers` — Chia thành words
- `models` — BPE, WordPiece, Unigram
- `trainers` — Train model trên corpus
- `post_processors` — Thêm special tokens, tạo type_ids
- `decoders` — Giải mã tokens → text

### Ví dụ: Xây dựng BERT tokenizer (WordPiece)

```python
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers, processors

tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))

# Normalization
tokenizer.normalizer = normalizers.Sequence([
    normalizers.NFD(),
    normalizers.Lowercase(),
    normalizers.StripAccents()
])

# Pre-tokenization
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# Training
special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]"]
trainer = trainers.WordPieceTrainer(vocab_size=25000, special_tokens=special_tokens)
tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

# Post-processing
cls_token_id = tokenizer.token_to_id("[CLS]")
sep_token_id = tokenizer.token_to_id("[SEP]")
tokenizer.post_processor = processors.TemplateProcessing(
    single=f"[CLS]:0 $A:0 [SEP]:0",
    pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
    special_tokens=[("[CLS]", cls_token_id), ("[SEP]", sep_token_id)],
)

# Decoder
tokenizer.decoder = decoders.WordPiece(prefix="##")
```

### Ví dụ: GPT-2 tokenizer (BPE + ByteLevel)

```python
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

trainer = trainers.BpeTrainer(vocab_size=25000, special_tokens=["<|endoftext|>"])
tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
tokenizer.decoder = decoders.ByteLevel()
```

### Ví dụ: XLNet tokenizer (Unigram + Metaspace)

```python
tokenizer = Tokenizer(models.Unigram())

tokenizer.normalizer = normalizers.Sequence([
    normalizers.Replace("``", '"'),
    normalizers.Replace("''", '"'),
    normalizers.NFKD(),
    normalizers.StripAccents(),
    normalizers.Replace(Regex(" {2,}"), " "),
])
tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()

special_tokens = ["<cls>", "<sep>", "<unk>", "<pad>", "<mask>", "<s>", "</s>"]
trainer = trainers.UnigramTrainer(vocab_size=25000, special_tokens=special_tokens, unk_token="<unk>")
tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)

# XLNet: <cls> ở cuối câu, type_id=2
tokenizer.post_processor = processors.TemplateProcessing(
    single="$A:0 <sep>:0 <cls>:2",
    pair="$A:0 <sep>:0 $B:1 <sep>:1 <cls>:2",
    special_tokens=[("<sep>", sep_token_id), ("<cls>", cls_token_id)],
)
tokenizer.decoder = decoders.Metaspace()
```

### Wrap tokenizer để dùng với 🤗 Transformers

```python
from transformers import PreTrainedTokenizerFast

wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    unk_token="[UNK]",
    pad_token="[PAD]",
    cls_token="[CLS]",
    sep_token="[SEP]",
    mask_token="[MASK]",
)

# Hoặc dùng class cụ thể:
from transformers import BertTokenizerFast
wrapped_tokenizer = BertTokenizerFast(tokenizer_object=tokenizer)
```

---

## Tóm tắt kiến thức cốt lõi

1. **Train tokenizer ≠ train model**: Tokenizer training là thống kê determinisitic, không phải học sâu.

2. **Chỉ train tokenizer mới khi**: Corpus của bạn khác biệt lớn so với corpus pretrain gốc (ngôn ngữ khác, domain khác).

3. **Fast tokenizer** (Rust) cung cấp:
   - Tokenization song song nhanh hơn
   - **Offset mapping**: map token ↔ vị trí ký tự gốc
   - `word_ids()`, `token_to_chars()`, `word_to_chars()`

4. **BPE**: Merge cặp phổ biến nhất. Dùng bởi GPT-2. Byte-level BPE không có UNK.

5. **WordPiece**: Merge dựa trên score `freq_pair/(freq_a×freq_b)`. Encoding: longest-match-first.

6. **Unigram**: Bắt đầu từ vocab lớn, prune token ít quan trọng. Encoding: Viterbi. Dùng SentencePiece.

7. **SentencePiece**: Không cần pre-tokenization, reversible, thay space bằng `▁`.

8. **QA pipeline xử lý context dài**: Chia thành chunks với stride overlap, dự đoán trên từng chunk, chọn kết quả tốt nhất.

9. **Pipeline tokenization**: Normalization → Pre-tokenization → Model → Post-processing.

10. **Xây dựng tokenizer tùy chỉnh**: Dùng các block `normalizers`, `pre_tokenizers`, `models`, `trainers`, `post_processors`, `decoders` của thư viện 🤗 Tokenizers, rồi wrap vào `PreTrainedTokenizerFast`.