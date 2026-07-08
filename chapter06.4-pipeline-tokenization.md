# Giải thích chi tiết Pipeline Tokenization

```
Raw text → Normalization → Pre-tokenization → Model → Post-processing → Tokens
```

Đây là 4 bước chính mà một tokenizer thực hiện để biến văn bản thô thành chuỗi token ID. Hãy đi qua từng bước một cách chi tiết.

---

## 1. Normalization (Chuẩn hóa)

**Mục đích:** Làm sạch văn bản đầu vào, đưa về một định dạng thống nhất trước khi xử lý tiếp.

**Các thao tác điển hình:**

### a) Lowercasing (chuyển về chữ thường)
"Hello World" → "hello world"

### b) Loại bỏ dấu (Strip Accents)
"Héllò" → "Hello"

### c) Chuẩn hóa Unicode (NFD, NFKD, NFC, NFKC)
Ví dụ ký tự `é` có thể biểu diễn bằng 1 Unicode code point (`U+00E9`) hoặc 2 code point (`e` + combining acute accent `U+0301`). Normalization đảm bảo mọi ký tự được biểu diễn nhất quán.

### d) Xóa/xử lý khoảng trắng thừa
"hello   world" → "hello world"

### e) Thay thế ký tự đặc biệt
Ví dụ XLNet thay ```` `` ```` và `''` thành `"`.

**Code minh họa:**

```python
from transformers import AutoTokenizer

# BERT uncased: lowercase + strip accents
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenizer.backend_tokenizer.normalizer.normalize_str("Héllò hôw are ü?")
# Kết quả: 'hello how are u?'

# GPT-2: không dùng normalizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")
print(type(tokenizer.backend_tokenizer.normalizer))
# Kết quả: <class 'tokenizers.normalizers.BertNormalizer'> (nhưng thực tế GPT-2 không normalize)
```

```python
# Xây dựng normalization thủ công
from tokenizers import normalizers

tokenizer.normalizer = normalizers.Sequence([
    normalizers.NFD(),           # Phân tách ký tự tổ hợp (é → e + ´)
    normalizers.Lowercase(),     # Chuyển về chữ thường
    normalizers.StripAccents()   # Bỏ dấu (e + ´ → e)
])
```

> **Lưu ý quan trọng:** Cần NFD trước StripAccents, vì StripAccents chỉ nhận diện được dấu khi nó ở dạng combining character riêng biệt.

---

## 2. Pre-tokenization (Tiền tokenization)

**Mục đích:** Chia văn bản thành các "từ" (word-level units). Đây là bước trung gian — chưa phải token cuối cùng, mà là ranh giới để model subword hoạt động bên trong từng từ.

**Nguyên tắc:** Model subword sẽ không bao giờ tạo ra token vượt qua ranh giới của pre-tokenization. Ví dụ: nếu pre-tokenizer tách "hello world" thành `["hello", "world"]`, thì tokenizer sẽ không tạo ra token `"oworld"` hay `"hellow"`.

### Các chiến lược pre-tokenization khác nhau:

#### BERT style: Tách trên whitespace + punctuation

```python
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str("Hello, how are  you?")
```

Kết quả:
```python
[('Hello', (0, 5)), (',', (5, 6)), ('how', (7, 10)), ('are', (11, 14)), ('you', (16, 19)), ('?', (19, 20))]
```

**Đặc điểm:**
- Tách từng dấu câu thành token riêng
- **Bỏ qua double space** — hai dấu cách giữa "are" và "you" bị rút gọn
- Tokenization **không reversible** (không thể phục hồi chính xác text gốc)

#### GPT-2 style: ByteLevel — tách whitespace + punctuation, GIỮ space

```python
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str("Hello, how are  you?")
```

Kết quả:
```python
[('Hello', (0, 5)), (',', (5, 6)), ('Ġhow', (6, 10)), ('Ġare', (10, 14)), ('Ġ', (14, 15)), ('Ġyou', (15, 19)), ('?', (19, 20))]
```

**Đặc điểm:**
- Space được giữ lại với ký hiệu `Ġ` đứng trước từ tiếp theo
- **Double space được giữ** (xuất hiện `Ġ` riêng lẻ)
- Byte-level encoding: làm việc ở mức byte thay vì ký tự Unicode → không bao giờ gặp UNK

#### SentencePiece / T5 style (Metaspace): Chỉ tách trên whitespace, GIỮ space, thêm space ở đầu

```python
tokenizer = AutoTokenizer.from_pretrained("t5-small")
tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str("Hello, how are  you?")
```

Kết quả:
```python
[('▁Hello,', (0, 6)), ('▁how', (7, 10)), ('▁are', (11, 14)), ('▁you?', (16, 20))]
```

**Đặc điểm:**
- Dấu câu **không bị tách riêng** (`,` nằm trong `▁Hello,`)
- Space được thay bằng `▁`
- **Luôn thêm `▁` ở đầu câu**
- Double space bị bỏ qua
- **Reversible**: decode = nối tokens + replace `▁` bằng space

### Tóm tắt khác biệt:

| | Tách space? | Tách punctuation? | Giữ double space? | Reversible? |
|---|---|---|---|---|
| **BERT** | ✅ | ✅ | ❌ | ❌ |
| **GPT-2** | ✅ | ✅ | ✅ | ✅ |
| **T5/SentencePiece** | ✅ | ❌ | ❌ | ✅ |

### Composing pre-tokenizers

Có thể kết hợp nhiều pre-tokenizer:

```python
pre_tokenizer = pre_tokenizers.Sequence([
    pre_tokenizers.WhitespaceSplit(),  # Tách theo space trước
    pre_tokenizers.Punctuation()       # Sau đó tách punctuation
])
pre_tokenizer.pre_tokenize_str("Let's test!")
# [('Let', (0,3)), ("'", (3,4)), ('s', (4,5)), ('test', (6,10)), ('!', (10,11))]
```

---

## 3. Model (Subword Tokenization)

**Mục đích:** Với mỗi "từ" từ bước pre-tokenization, model subword chia nhỏ thành các token thực sự dựa trên vocabulary đã được train.

Đây là **trái tim** của tokenizer. Có 3 thuật toán chính:

### BPE (Byte-Pair Encoding)

**Train:** Học các merge rules bằng cách merge cặp token xuất hiện nhiều nhất.

**Encode:** Áp dụng các merge rules theo đúng thứ tự đã học.

```python
# Ví dụ merge rules đã học:
# ("u","g") → "ug"
# ("u","n") → "un"
# ("h","ug") → "hug"

# Từ "hugs" được encode:
# Ban đầu:  ["h", "u", "g", "s"]
# Merge 1:  ["h", "ug", "s"]        (u,g → ug)
# Merge 2:  không merge (un)
# Merge 3:  ["hug", "s"]            (h,ug → hug)
# Kết quả:  ["hug", "s"]
```

### WordPiece

**Train:** Merge cặp có score cao nhất: `score = freq_pair / (freq_a × freq_b)`. Ưu tiên merge cặp mà mỗi phần riêng lẻ ít phổ biến.

**Encode:** Tìm subword dài nhất có trong vocab tính từ đầu từ, rồi lặp lại cho phần còn lại (longest-match-first, greedy).

```python
# Vocab: ["hug", "##s", "##g", "##u", "##gs"]
# Từ "hugs":
# - "hug" có trong vocab → token đầu: "hug"
# - Còn lại "##s" → có trong vocab → token thứ hai: "##s"
# Kết quả: ["hug", "##s"]

# Từ "bugs" (b không đầu từ):
# - "b" có trong vocab → token: "b"
# - Còn "##ugs": "##u" có trong vocab → token: "##u"
# - Còn "##gs": có trong vocab → token: "##gs"
# Kết quả: ["b", "##u", "##gs"]

# Nếu không tìm thấy subword nào → cả từ thành [UNK]
```

Prefix `##` đánh dấu token này là phần tiếp theo của từ (không đứng đầu).

### Unigram

**Train:** Bắt đầu từ vocab lớn, prune token ít ảnh hưởng đến loss. Mỗi token có một score.

**Encode:** Viterbi algorithm — tìm đường đi có xác suất (tích các score) cao nhất trong đồ thị tất cả phân đoạn có thể.

```python
# Vocab với score: "hug": 0.07, "p": 0.08, "u": 0.17, "g": 0.09, "pu": 0.08, ...
# Từ "pug":
# - ["p","u","g"]: P = 0.08 × 0.17 × 0.09 = 0.00122
# - ["p","ug"]:    P = 0.08 × 0.09 = 0.0072
# - ["pu","g"]:    P = 0.08 × 0.09 = 0.0072
# Kết quả: ["p", "ug"] hoặc ["pu", "g"] (score bằng nhau)
```

---

## 4. Post-processing (Hậu xử lý)

**Mục đích:** Thêm special tokens, tạo token type IDs, attention mask, padding, truncation.

Đây là bước cuối cùng trước khi đưa token vào model. Các công việc chính:

### a) Thêm special tokens

Mỗi model có bộ special token riêng:

| Model | Special tokens |
|-------|---------------|
| BERT | `[CLS]` ... `[SEP]` |
| GPT-2 | `<\|endoftext\|>` |
| XLNet | `...<sep> <cls>` (`<cls>` ở cuối!) |

**Template BERT:**
```
Single:  [CLS] $A [SEP]
Pair:    [CLS] $A [SEP] $B [SEP]
```

**Template XLNet (đặc biệt):**
```
Single:  $A <sep> <cls>
Pair:    $A <sep> $B <sep> <cls>
```
(`<cls>` ở cuối, padding bên trái)

### b) Token type IDs (segment IDs)

Đánh dấu token thuộc câu nào:
- `0` cho câu thứ nhất (hoặc single sentence)
- `1` cho câu thứ hai
- `2` cho special token `<cls>` của XLNet

```python
# BERT encode cặp câu
encoding = tokenizer("Hello.", "How are you?")
encoding.type_ids
# [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
#  ^CLS  ^Hello ^. ^SEP  ^How ^are ^you ^? ^SEP
```

### c) Attention mask

`1` cho token thật, `0` cho padding token. Đảm bảo model biết bỏ qua padding.

### d) Padding & Truncation

- **Padding:** Đệm thêm token `[PAD]` để tất cả sequence trong batch có cùng độ dài
- **Truncation:** Cắt bớt sequence quá dài. Có thể chọn chiến lược: `only_first`, `only_second`, `longest_first`

### e) Offset mapping (chỉ fast tokenizer)

Post-processor có thể điều chỉnh offset để ánh xạ chính xác từ token về text gốc.

---

## Sơ đồ tổng quan toàn bộ pipeline

```
INPUT: "Hello, how are you?"
│
▼ Normalization
│  "hello, how are you?"          ← lowercase, strip accents, clean text
│
▼ Pre-tokenization  
│  ["hello", ",", "how", "are", "you", "?"]   ← split thành word-level
│
▼ Model (e.g. BPE)
│  ["hello", ",", "how", "are", "you", "?"]   ← từ nào trong vocab thì giữ nguyên
│  ["hell", "##o", ",", "how", "are", "you", "?"]   ← từ ngoài vocab thì chia subword
│
▼ Post-processing
│  ["[CLS]", "hell", "##o", ",", "how", "are", "you", "?", "[SEP]"]
│
▼ Token IDs
│  [101, 2081, 2075, 1010, 2129, 2024, 2017, 1029, 102]
│
OUTPUT: tokens (dùng cho model)
```

### Tóm tắt trách nhiệm từng bước

| Bước | Câu hỏi trả lời | Ví dụ input → output |
|------|-----------------|----------------------|
| **Normalization** | Text có sạch và nhất quán không? | `"Héllò"` → `"hello"` |
| **Pre-tokenization** | Đâu là ranh giới "từ"? | `"hello world"` → `["hello", "world"]` |
| **Model** | Mỗi từ nên chia thành những token nào? | `["hello"]` → `["hell", "##o"]` |
| **Post-processing** | Cần thêm special token gì? Padding ra sao? | `["hell","##o"]` → `["[CLS]","hell","##o","[SEP]"]` |