# Giải thích chi tiết: Gom nhóm Entity trong NER

---

## 1. Bối cảnh: Mô hình NER trả về gì?

Khi một câu được tokenize, BERT chia từ thành các subword. Ví dụ:

```
"My name is Sylvain and I work at Hugging Face in Brooklyn."
```

BERT tokenizer tạo ra 19 tokens, trong đó "Sylvain" bị tách thành 4 subword: `S`, `##yl`, `##va`, `##in`. Mô hình NER gán nhãn cho **từng token**, nên ta nhận được:

| Token ID | Token | Nhãn dự đoán |
|----------|-------|-------------|
| 4 | `S` | `I-PER` |
| 5 | `##yl` | `I-PER` |
| 6 | `##va` | `I-PER` |
| 7 | `##in` | `I-PER` |
| ... | ... | ... |
| 12 | `Hu` | `I-ORG` |
| 13 | `##gging` | `I-ORG` |
| 14 | `Face` | `I-ORG` |
| ... | ... | ... |
| 16 | `Brooklyn` | `I-LOC` |

Nhưng người dùng cuối không quan tâm đến subword — họ muốn biết **"Sylvain" là một Person**, **"Hugging Face" là một Organization**. Vậy phải **gộp các token subword thành entity hoàn chỉnh**.

---

## 2. Các loại nhãn BIO

Format nhãn NER phổ biến là **BIO** (Beginning–Inside–Outside):

| Nhãn | Ý nghĩa |
|------|--------|
| `O` | Outside — token này không thuộc entity nào |
| `B-PER` | **Beginning** — token đầu tiên của một Person entity |
| `I-PER` | **Inside** — token bên trong một Person entity |
| `B-ORG` | **Beginning** — token đầu tiên của Organization |
| `I-ORG` | **Inside** — token bên trong Organization |
| `B-LOC` / `I-LOC` | Tương tự cho Location |
| `B-MISC` / `I-MISC` | Miscellaneous |

### Ý tưởng gom nhóm

Các token có **cùng loại entity và liên tiếp nhau** sẽ được gộp thành một entity. Điểm bắt đầu là token `B-XXX`, các token sau là `I-XXX`.

```
B-PER  I-PER  I-PER   →   gộp thành 1 entity PER
B-ORG  I-ORG          →   gộp thành 1 entity ORG
B-LOC                 →   1 entity LOC
```

---

## 3. IOB1 vs IOB2 — Sự khác biệt quan trọng

Có **hai biến thể** của format BIO, và điều này ảnh hưởng đến logic gom nhóm:

### IOB2 (phổ biến hơn)
`B-XXX` luôn được dùng cho **token đầu tiên** của mọi entity.

```
Sylvain →  B-PER
```

### IOB1 (dùng trong dataset CoNLL-03 — checkpoint `dbmdz/bert-large-cased-finetuned-conll03-english`)
`B-XXX` **chỉ** được dùng để phân biệt hai entity cùng loại **đứng liền kề nhau**. Token đầu tiên của entity vẫn dùng `I-XXX`.

```
Sylvain →  I-PER   (vì không có entity PER nào đứng ngay trước nó)
```

**Ví dụ so sánh:**

> "John Smith and Jane Doe went to Paris."

Giả sử "John Smith" và "Jane Doe" là 2 PER đứng cạnh nhau (cách bởi "and"):

| Token | IOB1 | IOB2 |
|-------|------|------|
| John | `I-PER` | `B-PER` |
| Smith | `I-PER` | `I-PER` |
| and | `O` | `O` |
| Jane | `I-PER` | `B-PER` |
| Doe | `I-PER` | `I-PER` |

→ IOB2 dùng `B-` mỗi khi bắt đầu entity mới. IOB1 chỉ dùng `B-` khi cần "ngắt" — tức là trong IOB1, nếu ta thấy `B-PER` thì chắc chắn ngay trước đó có một token khác cũng là `I-PER` (cùng loại, đứng liền kề).

---

## 4. Logic gom nhóm tổng quát (xử lý cả IOB1 và IOB2)

Vì mô hình có thể dùng IOB1 hoặc IOB2, ta cần logic **chịu được cả hai format**:

> **Gom các token liên tiếp có nhãn `B-XXX` hoặc `I-XXX` (cùng loại XXX). Dừng khi gặp `O`, nhãn loại khác, hoặc `B-XXX` cùng loại (đánh dấu bắt đầu entity mới ngay sau entity cũ).**

Code minh họa:

```python
import numpy as np

results = []
inputs_with_offsets = tokenizer(example, return_offsets_mapping=True)
tokens = inputs_with_offsets.tokens()
offsets = inputs_with_offsets["offset_mapping"]

idx = 0
while idx < len(predictions):
    pred = predictions[idx]
    label = model.config.id2label[pred]   # vd: "I-PER", "B-ORG", "O"
    
    if label != "O":
        # Bỏ tiền tố B- hoặc I-, chỉ giữ loại entity (PER, ORG, LOC, MISC)
        entity_type = label[2:]  # "I-PER" → "PER"
        start_char, _ = offsets[idx]  # vị trí bắt đầu trong câu gốc

        # Gom tất cả token I-XXX (và B-XXX đầu tiên) liên tiếp
        all_scores = []
        while (
            idx < len(predictions)
            and model.config.id2label[predictions[idx]] in [f"B-{entity_type}", f"I-{entity_type}"]
        ):
            # Kiểm tra: nếu gặp B-XXX cùng loại và đây KHÔNG phải token đầu tiên → dừng
            # (vì B-XXX đánh dấu entity mới cùng loại đứng ngay sau)
            current_label = model.config.id2label[predictions[idx]]
            if current_label == f"B-{entity_type}" and len(all_scores) > 0:
                break
            
            all_scores.append(probabilities[idx][predictions[idx]])
            _, end_char = offsets[idx]
            idx += 1

        score = np.mean(all_scores).item()
        word = example[start_char:end_char]  # trích xuất text từ offset
        results.append({
            "entity_group": entity_type,
            "score": score,
            "word": word,
            "start": start_char,
            "end": end_char,
        })
    idx += 1
```

---

## 5. Tại sao dùng offset thay vì ghép `##` thủ công?

Có thể bạn nghĩ: "Cứ ghép token lại, bỏ `##`, thêm space trước token không có `##` là xong mà?" Cách đó chỉ hoạt động cho **BERT WordPiece**. Nhưng offset mapping hoạt động với **mọi loại tokenizer** (BPE, Unigram, SentencePiece, ...).

### So sánh hai cách

**Cách thủ công (chỉ cho BERT WordPiece):**

```python
# Rất fragile, chỉ đúng cho WordPiece
def merge_tokens(tokens):
    result = ""
    for tok in tokens:
        if tok.startswith("##"):
            result += tok[2:]  # bỏ ##
        else:
            result += " " + tok
    return result.strip()

merge_tokens(["Hu", "##gging", "Face"])  # "Hugging Face" ✅
merge_tokens(["Hugg", "##ing"])           # "Hugging" ✅
```

**Cách dùng offset mapping (cho mọi tokenizer):**

```python
start_char, _ = offset_of_first_token   # offset của token "Hu" → (33, 35)
_, end_char = offset_of_last_token       # offset của token "Face" → (41, 45)
example[start_char:end_char]             # "Hugging Face" ✅
```

Offset cho biết **vị trí chính xác trong text gốc** của mỗi token, bao gồm cả khoảng trắng nếu có (GPT-2 dùng `Ġ` cho space, offset của `Ġtest` là `(5, 10)` — bao gồm cả space trước từ).

---

## 6. Ví dụ minh họa toàn bộ quá trình

```python
example = "My name is Sylvain and I work at Hugging Face in Brooklyn."

# Sau khi model dự đoán, ta có:
# Tokens: ['[CLS]', 'My', 'name', 'is', 'S', '##yl', '##va', '##in', 
#           'and', 'I', 'work', 'at', 'Hu', '##gging', 'Face', 'in', 'Brooklyn', '.', '[SEP]']
# Predictions: [0, 0, 0, 0, 4, 4, 4, 4, 0, 0, 0, 0, 6, 6, 6, 0, 8, 0, 0]
#                                   ↑I-PER↑              ↑I-ORG↑      ↑I-LOC↑
# offsets[4]  = (11, 12)  # token "S"
# offsets[7]  = (16, 18)  # token "##in"
# offsets[12] = (33, 35)  # token "Hu"
# offsets[14] = (41, 45)  # token "Face"
# offsets[16] = (49, 57)  # token "Brooklyn"

# Entity PER:
example[11:18]  # "Sylvain"

# Entity ORG:
example[33:45]  # "Hugging Face"

# Entity LOC:
example[49:57]  # "Brooklyn"
```

---

## 7. Tóm tắt

| Vấn đề | Giải pháp |
|--------|----------|
| Mô hình trả nhãn cho từng subword | Gom các subword cùng label và liền kề |
| Cần biết đâu là token bắt đầu entity | Dùng `B-XXX` (IOB2) hoặc heuristic (IOB1) |
| Cần lấy text gốc không có `##` | Dùng offset mapping (`example[start:end]`) |
| Cần hoạt động với mọi tokenizer | Offset mapping là giải pháp universal |

**Điểm mấu chốt:** Offset mapping là cây cầu nối giữa thế giới token (subword) và thế giới text gốc (character-level). Nhờ nó, việc gom nhóm entity trở nên đơn giản: chỉ cần lấy span từ offset của token đầu tiên đến offset của token cuối cùng trong nhóm.