# Chương 5: Thư viện 🤗 Datasets – Từ Cơ Bản đến Nâng Cao

## 1. Tổng quan

Chương này đi sâu vào thư viện `🤗 Datasets`, giải quyết các câu hỏi:
- Làm gì khi dataset **không có sẵn trên Hugging Face Hub**?
- Làm sao để **cắt, lọc, biến đổi** dữ liệu linh hoạt?
- Xử lý dataset **quá lớn, vượt RAM** của máy như thế nào?
- **Memory mapping** và **Apache Arrow** là gì?
- Làm sao để **tạo dataset của riêng bạn** và đẩy lên Hub?

---

## 2. Load dataset từ local và remote (không cần Hub)

### Các định dạng được hỗ trợ

`load_dataset()` có thể load trực tiếp từ file local/remote qua tham số `data_files`:

| Định dạng     | Script load | Ví dụ |
|:-------------:|:-----------:|:------|
| CSV & TSV     | `csv`       | `load_dataset("csv", data_files="file.csv")` |
| Text files    | `text`      | `load_dataset("text", data_files="file.txt")` |
| JSON & JSONL  | `json`       | `load_dataset("json", data_files="file.jsonl")` |
| Pickle (Pandas)| `pandas`   | `load_dataset("pandas", data_files="df.pkl")` |

### Load local dataset

```python
from datasets import load_dataset

# Load JSON với field="data" (dạng nested)
squad_it_dataset = load_dataset("json", data_files="SQuAD_it-train.json", field="data")

# Load nhiều splits bằng dictionary
data_files = {"train": "train.json", "test": "test.json"}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
```

Mặc định, load local file tạo ra `DatasetDict` với split `train`.

### Load remote dataset

Chỉ cần trỏ `data_files` tới URL thay vì đường dẫn local:

```python
url = "https://github.com/.../raw/master/"
data_files = {
    "train": url + "SQuAD_it-train.json.gz",
    "test": url + "SQuAD_it-test.json.gz",
}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
```

> **Quan trọng:** `data_files` hỗ trợ tự động giải nén (gzip, zip, tar). Có thể dùng glob pattern (`*.json`), list, hoặc dict ánh xạ split→file.

---

## 3. "Slicing and Dicing" – Làm sạch và biến đổi dữ liệu

### Các thao tác cơ bản

| Hàm | Công dụng |
|:---|:---|
| `Dataset.shuffle(seed=...)` | Xáo trộn dữ liệu |
| `Dataset.select(indices)` | Chọn hàng theo index |
| `Dataset.unique(column)` | Lấy giá trị duy nhất trong cột |
| `Dataset.rename_column()` | Đổi tên cột |
| `Dataset.sort(column)` | Sắp xếp theo cột |
| `Dataset.train_test_split()` | Chia train/test |

```python
# Random sample 1000 dòng
drug_sample = drug_dataset["train"].shuffle(seed=42).select(range(1000))

# Kiểm tra unique values
for split in drug_dataset.keys():
    assert len(drug_dataset[split]) == len(drug_dataset[split].unique("Unnamed: 0"))

# Đổi tên cột
drug_dataset = drug_dataset.rename_column("Unnamed: 0", "patient_id")
```

### `Dataset.map()` – Hàm biến đổi mạnh mẽ nhất

```python
# Lowercase cột condition
def lowercase_condition(example):
    return {"condition": example["condition"].lower()}

drug_dataset = drug_dataset.map(lowercase_condition)
```

### `Dataset.filter()` – Lọc dữ liệu

```python
# Lọc bỏ None dùng lambda
drug_dataset = drug_dataset.filter(lambda x: x["condition"] is not None)

# Lọc review ngắn
drug_dataset = drug_dataset.filter(lambda x: x["review_length"] > 30)
```

### Tạo cột mới

```python
# Cách 1: Dùng map() trả về key mới
def compute_review_length(example):
    return {"review_length": len(example["review"].split())}
drug_dataset = drug_dataset.map(compute_review_length)

# Cách 2: Dùng Dataset.add_column() (nhận list/NumPy array)
```

### Siêu năng lực của `map()`: `batched=True`

Khi `batched=True`, hàm nhận vào **batch dictionary** (mỗi value là list thay vì scalar), giúp tăng tốc đáng kể:

```python
# Unescape HTML nhanh hơn với batched + list comprehension
drug_dataset = drug_dataset.map(
    lambda x: {"review": [html.unescape(o) for o in x["review"]]},
    batched=True
)
```

**So sánh hiệu năng tokenizer:**

| Tùy chọn | Fast tokenizer | Slow tokenizer |
|:---|:---:|:---:|
| `batched=True` | 10.8s | 4ph41s |
| `batched=False` | 59.2s | 5ph3s |
| `batched=True, num_proc=8` | 6.52s | 41.3s |

> **Tại sao Fast Tokenizer nhanh hơn nhiều?** Code tokenizer chạy bằng **Rust**, cho phép parallelize trên batch. `num_proc` dùng Python multiprocessing, không hiệu quả bằng Rust nhưng vẫn giúp ích cho slow tokenizer.

### `return_overflowing_tokens=True` – Chia văn bản dài thành nhiều chunk

```python
def tokenize_and_split(examples):
    return tokenizer(
        examples["review"],
        truncation=True,
        max_length=128,
        return_overflowing_tokens=True,
    )

# CẢNH BÁO: Khi return_overflowing_tokens=True, số output feature nhiều hơn input
# → Phải remove_columns hoặc map ngược lại bằng overflow_to_sample_mapping

tokenized_dataset = drug_dataset.map(
    tokenize_and_split, batched=True,
    remove_columns=drug_dataset["train"].column_names  # Quan trọng!
)
```

Giữ nguyên cột cũ bằng `overflow_to_sample_mapping`:
```python
def tokenize_and_split(examples):
    result = tokenizer(examples["review"], truncation=True, max_length=128,
                       return_overflowing_tokens=True)
    sample_map = result.pop("overflow_to_sample_mapping")
    for key, values in examples.items():
        result[key] = [values[i] for i in sample_map]  # Lặp lại giá trị cũ
    return result
```

### Tương tác với Pandas

```python
# Chuyển output format sang Pandas
drug_dataset.set_format("pandas")
train_df = drug_dataset["train"][:]  # Lấy toàn bộ DataFrame

# Làm việc với Pandas
frequencies = train_df["condition"].value_counts().to_frame().reset_index()

# Quay lại Arrow format
drug_dataset.reset_format()

# Tạo Dataset từ Pandas DataFrame
from datasets import Dataset
freq_dataset = Dataset.from_pandas(frequencies)
```

> **Lưu ý:** `set_format()` chỉ thay đổi output format của `__getitem__()`, không thay đổi underlying data format (Apache Arrow).

### Tạo validation set

```python
drug_dataset_clean = drug_dataset["train"].train_test_split(train_size=0.8, seed=42)
drug_dataset_clean["validation"] = drug_dataset_clean.pop("test")
drug_dataset_clean["test"] = drug_dataset["test"]
```

### Lưu dataset ra disk

| Định dạng | Hàm |
|:---|:---|
| Arrow | `Dataset.save_to_disk()` |
| CSV | `Dataset.to_csv()` |
| JSON | `Dataset.to_json()` |

```python
# Lưu Arrow format
drug_dataset_clean.save_to_disk("drug-reviews")

# Load lại
from datasets import load_from_disk
drug_reloaded = load_from_disk("drug-reviews")

# Lưu từng split ra JSON Lines
for split, dataset in drug_dataset_clean.items():
    dataset.to_json(f"drug-reviews-{split}.jsonl")
```

---

## 4. Big Data – Memory Mapping và Streaming

### Memory Mapping

🤗 Datasets dùng **Apache Arrow** làm định dạng lưu trữ. Dataset được xử lý như **memory-mapped file** – ánh xạ giữa RAM và filesystem, cho phép truy cập dataset **mà không cần load toàn bộ vào RAM**.

**Ví dụ thực tế:** Dataset PubMed Abstracts: **19.54 GB** trên disk nhưng chỉ tốn khoảng **5.6 GB RAM** để load và truy cập. Tốc độ duyệt đạt ~0.3 GB/s.

> **So sánh với Pandas:** Quy tắc ngón tay cái của Pandas là cần RAM gấp **5-10 lần** kích thước dataset. 🤗 Datasets giải quyết triệt để vấn đề này nhờ memory mapping + Apache Arrow.

### Streaming Datasets

Khi dataset **quá lớn không đủ chỗ trên ổ cứng**, dùng `streaming=True`:

```python
pubmed_streamed = load_dataset("json", data_files=data_files, split="train", streaming=True)
# Trả về IterableDataset (generator), không phải Dataset (container)

# Truy cập từng phần tử
next(iter(pubmed_streamed))

# Tokenize on-the-fly
tokenized = pubmed_streamed.map(lambda x: tokenizer(x["text"]))

# Shuffle với buffer
shuffled = pubmed_streamed.shuffle(buffer_size=10_000, seed=42)

# Lấy N phần tử đầu / bỏ qua N phần tử
head = pubmed_streamed.take(5)
train = shuffled.skip(1000)
val = shuffled.take(1000)
```

### `interleave_datasets()` – Kết hợp nhiều streaming dataset

```python
from datasets import interleave_datasets
combined = interleave_datasets([pubmed_streamed, law_streamed])
# Xen kẽ từng phần tử từ mỗi dataset
```

### Load toàn bộ The Pile (825 GB)

```python
base_url = "https://the-eye.eu/public/AI/pile/"
data_files = {
    "train": [base_url + "train/" + f"{idx:02d}.jsonl.zst" for idx in range(30)],
    "validation": base_url + "val.jsonl.zst",
    "test": base_url + "test.jsonl.zst",
}
pile_dataset = load_dataset("json", data_files=data_files, streaming=True)
```

---

## 5. Tạo dataset của riêng bạn

### Lấy dữ liệu từ GitHub Issues API

```python
import requests

url = "https://api.github.com/repos/huggingface/datasets/issues?page=1&per_page=100"
headers = {"Authorization": f"token {GITHUB_TOKEN}"}  # Cần token để tăng rate limit
response = requests.get(url)
issues = response.json()
```

> **Rate limit:** Unauthenticated = 60 req/h; có token = 5000 req/h.

### Làm sạch và augment

```python
# Phân biệt issue vs pull request
issues_dataset = issues_dataset.map(
    lambda x: {"is_pull_request": x["pull_request"] is not None}
)

# Lấy comments cho từng issue qua API
def get_comments(issue_number):
    url = f"https://api.github.com/repos/.../issues/{issue_number}/comments"
    response = requests.get(url, headers=headers)
    return [r["body"] for r in response.json()]

issues_with_comments = issues_dataset.map(
    lambda x: {"comments": get_comments(x["number"])}
)
```

### Đẩy dataset lên Hugging Face Hub

```python
from huggingface_hub import notebook_login
notebook_login()  # Hoặc huggingface-cli login

issues_with_comments.push_to_hub("github-issues")

# Người khác có thể load bằng
remote_dataset = load_dataset("lewtun/github-issues", split="train")
```

### Dataset Card

Tạo file `README.md` chứa metadata YAML (qua `datasets-tagging`) và mô tả chi tiết dataset: intended use, supported tasks, biases, licensing. Dataset card tốt giúp cộng đồng dễ tìm kiếm và đánh giá dataset.

---

## 6. Semantic Search với FAISS

### Ý tưởng

- Dùng Transformer model để tạo **embedding vector** cho mỗi document (issue title + body + comment).
- Dùng **FAISS** (Facebook AI Similarity Search) để index và tìm kiếm nhanh các vector gần nhất.
- Với query người dùng → embed → tìm nearest neighbors → trả về kết quả.

### Chuẩn bị dữ liệu

```python
# Lọc bỏ PR và issue không có comment
issues_dataset = issues_dataset.filter(
    lambda x: not x["is_pull_request"] and len(x["comments"]) > 0
)

# Giữ các cột cần thiết
issues_dataset = issues_dataset.remove_columns(columns_to_remove)

# "Explode" cột comments → mỗi dòng là 1 comment
# (Có thể dùng Pandas explode hoặc Dataset.map với batch)
```

### Tạo Embeddings

```python
from transformers import AutoTokenizer, AutoModel

model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt).to("cuda")

# CLS Pooling: lấy hidden state của token [CLS] làm vector đại diện
def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]

def get_embeddings(text_list):
    encoded = tokenizer(text_list, padding=True, truncation=True, return_tensors="pt")
    encoded = {k: v.to("cuda") for k, v in encoded.items()}
    output = model(**encoded)
    return cls_pooling(output)

# Map qua toàn bộ dataset
embeddings_dataset = comments_dataset.map(
    lambda x: {"embeddings": get_embeddings(x["text"]).detach().cpu().numpy()[0]}
)
```

> **Asymmetric Semantic Search:** Query ngắn + document dài → cần model phù hợp (vd: `multi-qa-mpnet-base-dot-v1`).

### FAISS Index và tìm kiếm

```python
# Tạo FAISS index trên cột embeddings
embeddings_dataset.add_faiss_index(column="embeddings")

# Embed query
question_embedding = get_embeddings([question]).cpu().detach().numpy()

# Tìm k nearest neighbors
scores, samples = embeddings_dataset.get_nearest_examples(
    "embeddings", question_embedding, k=5
)
```

---

## 7. Tổng kết – Những điều cốt lõi cần nhớ

| Kỹ năng | Công cụ chính |
|:---|:---|
| Load dataset từ mọi nguồn | `load_dataset()` với `data_files` (local path, URL, glob) |
| Làm sạch & biến đổi | `map()`, `filter()`, `set_format()` |
| Xử lý batch hiệu quả | `map(batched=True)` + fast tokenizer (Rust) |
| Làm việc với văn bản dài | `return_overflowing_tokens=True` + `overflow_to_sample_mapping` |
| Dataset khổng lồ | Memory mapping (Arrow) + Streaming |
| Tạo dataset | GitHub API → `Dataset.from_pandas()` → `push_to_hub()` |
| Semantic search | Embeddings (Transformer) + FAISS index |

---

## 8. Quiz – Một số câu hỏi kiểm tra

1. **`load_dataset()` load được từ đâu?** → Local, Hub, Remote server (tất cả đều đúng).
2. **Random sample 50 phần tử?** → `dataset.shuffle().select(range(50))`.
3. **Lọc pet tên bắt đầu bằng "L"?** → `pets_dataset.filter(lambda x: x['name'].startswith('L'))`.
4. **Memory mapping là gì?** → Ánh xạ giữa RAM và filesystem storage.
5. **Streaming dataset truy cập thế nào?** → `next(iter(dataset))` (vì là `IterableDataset`).
6. **Asymmetric semantic search?** → Query ngắn, document dài trả lời query.
7. **🤗 Datasets có dùng được cho audio/vision không?** → Có.