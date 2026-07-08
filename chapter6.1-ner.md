## NER (Named Entity Recognition)

**NER** (Nhận dạng thực thể có tên) là một bài toán NLP mà mục tiêu là **xác định và phân loại các thực thể có tên** trong văn bản — như tên người, tổ chức, địa điểm, ngày tháng, v.v.

Ví dụ với câu:

> *"My name is Sylvain and I work at Hugging Face in Brooklyn."*

NER sẽ gán nhãn cho từng từ/token:

| Token | Nhãn |
|-------|------|
| Sylvain | `PER` (Person) |
| Hugging Face | `ORG` (Organization) |
| Brooklyn | `LOC` (Location) |

### Các loại nhãn NER phổ biến

| Nhãn | Ý nghĩa |
|------|---------|
| `PER` | Person — tên người |
| `ORG` | Organization — tổ chức |
| `LOC` | Location — địa điểm |
| `MISC` | Miscellaneous — khác (sự kiện, sản phẩm,...) |

### Hai format đánh nhãn: BIO / IOB

Mỗi token có thể có prefix `B-` (Beginning) hoặc `I-` (Inside):

| Nhãn | Ý nghĩa |
|------|---------|
| `B-PER` | Token **đầu tiên** của một thực thể Person |
| `I-PER` | Token **bên trong** của thực thể Person |
| `O` | Không thuộc thực thể nào (Outside) |

Ví dụ: `"Sylvain"` có thể được tokenize thành 4 token `["S", "##yl", "##va", "##in"]`:

```
S      → B-PER (hoặc I-PER trong IOB1)
##yl   → I-PER
##va   → I-PER
##in   → I-PER
```

Sau đó nhóm lại: `Sylvain → PER`.

### Pipeline NER trong Hugging Face

```python
from transformers import pipeline

ner = pipeline("token-classification")
ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")
```

```python
# Output:
[{'entity': 'I-PER', 'word': 'S', ...},
 {'entity': 'I-PER', 'word': '##yl', ...},
 {'entity': 'I-ORG', 'word': 'Hugging', ...},
 {'entity': 'I-LOC', 'word': 'Brooklyn', ...}]
```

Có thể gom nhóm entity với `aggregation_strategy="simple"` để được kết quả gọn hơn:

```python
ner = pipeline("token-classification", aggregation_strategy="simple")
# Output: Sylvain → PER, Hugging Face → ORG, Brooklyn → LOC
```

### Vai trò của offset mapping

Fast tokenizer dùng **offset mapping** để biết mỗi token con đến từ vị trí ký tự nào trong câu gốc. Điều này rất quan trọng vì model hoạt động ở mức token, nhưng ta cần trả về vị trí chính xác của thực thể trong văn bản gốc (để hiển thị highlight, v.v.). Đây chính là nội dung được phân tích sâu trong Section 3 của Chapter 6 mà chúng ta vừa tổng hợp.