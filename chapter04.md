# Chương 4: Hugging Face Hub — Tổng hợp chi tiết

---

## 1. Giới thiệu về Hugging Face Hub

Hugging Face Hub (`huggingface.co`) là **nền tảng trung tâm** cho phép mọi người khám phá, sử dụng và đóng góp các mô hình cũng như dataset state-of-the-art. Hub có hơn **10.000 mô hình công khai**.

### Điểm quan trọng:
- **Không giới hạn thư viện**: Các mô hình trên Hub không chỉ giới hạn ở 🤗 Transformers hay NLP. Có mô hình từ Flair, AllenNLP (NLP), Asteroid, pyannote (speech), timm (vision), v.v.
- **Mỗi mô hình là một Git repository** → cho phép versioning và reproducibility (tái tạo được kết quả).
- **Inference API tự động**: Khi chia sẻ mô hình lên Hub, Hub tự động triển khai Inference API cho mô hình đó. Người dùng có thể test trực tiếp trên trang mô hình.
- **Miễn phí** với mô hình public. Có paid plans cho mô hình private.
- **Cần tài khoản** `huggingface.co` để tạo và quản lý repository.

---

## 2. Sử dụng Pretrained Models (Mô hình đã huấn luyện sẵn)

### 2.1. Sử dụng qua `pipeline()`

Cách đơn giản nhất để dùng một pretrained model:

```python
from transformers import pipeline

camembert_fill_mask = pipeline("fill-mask", model="camembert-base")
results = camembert_fill_mask("Le camembert est <mask> :)")
# Kết quả: délicieux, excellent, succulent, meilleur, parfait...
```

> ⚠️ **Nguyên tắc quan trọng**: Checkpoint phải phù hợp với task. Ví dụ `camembert-base` dùng cho `fill-mask` thì đúng, nhưng nếu dùng cho `text-classification` thì kết quả sẽ vô nghĩa vì head của model không phù hợp. Luôn dùng **task selector** trên Hub để chọn checkpoint phù hợp.

### 2.2. Sử dụng qua `Auto*` classes (Khuyến nghị)

```python
from transformers import AutoTokenizer, AutoModelForMaskedLM

tokenizer = AutoTokenizer.from_pretrained("camembert-base")
model = AutoModelForMaskedLM.from_pretrained("camembert-base")
```

**Lợi ích của `Auto*` classes**:
- **Architecture-agnostic** (không phụ thuộc kiến trúc cụ thể)
- Dễ dàng đổi checkpoint mà không cần đổi code
- Ví dụ: `AutoModelForMaskedLM.from_pretrained("bert-base-uncased")` vẫn hoạt động mà không cần đổi `CamembertForMaskedLM` → `BertForMaskedLM`

> ⚠️ Nếu dùng lớp cụ thể như `CamembertForMaskedLM`, bạn bị giới hạn chỉ load được checkpoint của kiến trúc CamemBERT.

- Với PyTorch: `AutoTokenizer`, `AutoModel`, `AutoModelForSequenceClassification`, `AutoModelForMaskedLM`, v.v.
- Với TensorFlow: `TFAutoModel`, `TFAutoModelForMaskedLM`, v.v. (prefix `TF`)

### 2.3. Kiểm tra model card

Luôn kiểm tra **model card** của model để biết:
- Model được huấn luyện như thế nào, trên dataset nào
- Giới hạn và bias của model
- Cách sử dụng

---

## 3. Chia sẻ Pretrained Models lên Hub

Có **3 cách** để tạo model repository và upload files lên Hub:

| Cách | Mô tả |
|------|-------|
| `push_to_hub` API | Tích hợp trong 🤗 Transformers, đơn giản nhất |
| `huggingface_hub` library | Package Python riêng, API thấp hơn, linh hoạt hơn |
| Web interface | Dùng giao diện web trực tiếp trên `huggingface.co` |

### 3.1. Xác thực (Authentication)

Trước khi upload, cần đăng nhập để tạo token:

```bash
# Trong terminal:
huggingface-cli login
```

```python
# Trong notebook:
from huggingface_hub import notebook_login
notebook_login()
```

Token được lưu trong cache folder.

### 3.2. Cách 1: Sử dụng `push_to_hub` API

#### a) Với PyTorch + `Trainer`:

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    "bert-finetuned-mrpc",
    save_strategy="epoch",
    push_to_hub=True  # Tự động upload sau mỗi lần save
)

# Khi gọi trainer.train(), model được upload sau mỗi epoch
# Kết thúc training, gọi thêm:
trainer.push_to_hub()  # Upload phiên bản cuối + tạo model card tự động
```

- Để upload vào organization: `hub_model_id="my_organization/my_repo_name"`
- Model card tự động được sinh ra, chứa metadata như hyperparameters, evaluation results.

#### b) Với TensorFlow / Keras:

```python
from transformers import PushToHubCallback

callback = PushToHubCallback(
    "bert-finetuned-mrpc",
    save_strategy="epoch",
    tokenizer=tokenizer
)

model.fit(..., callbacks=[callback])
```

#### c) Gọi trực tiếp `push_to_hub()` trên model/tokenizer:

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

checkpoint = "camembert-base"
model = AutoModelForMaskedLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Sau khi train/fine-tune...
model.push_to_hub("dummy-model")       # Tạo repo + upload model
tokenizer.push_to_hub("dummy-model")   # Upload tokenizer vào cùng repo
```

- Upload vào organization:
```python
tokenizer.push_to_hub("dummy-model", organization="huggingface")
```

- Dùng token riêng:
```python
tokenizer.push_to_hub("dummy-model", organization="huggingface", use_auth_token="<TOKEN>")
```

### 3.3. Cách 2: Sử dụng `huggingface_hub` Python library

Package `huggingface_hub` cung cấp API thấp hơn, linh hoạt:

```python
from huggingface_hub import (
    login, logout, whoami,            # User management
    create_repo, delete_repo,         # Repository management
    update_repo_visibility,
    list_models, list_datasets,       # Query Hub
    list_repo_files,
    upload_file, delete_file,         # File operations
    Repository,                       # Local git-like repo management
)
```

#### a) Tạo repository:

```python
from huggingface_hub import create_repo

create_repo("dummy-model")                           # Tạo trong namespace cá nhân
create_repo("dummy-model", organization="huggingface")  # Tạo trong organization
create_repo("dummy-model", private=True)              # Repository private
```

#### b) Upload file (cách đơn giản nhất):

`upload_file` dùng HTTP POST, **không cần git/git-lfs**, nhưng giới hạn file < 5GB:

```python
from huggingface_hub import upload_file

upload_file(
    "<path_to_file>/config.json",
    path_in_repo="config.json",
    repo_id="<namespace>/dummy-model",
)
```

#### c) Sử dụng `Repository` class:

`Repository` quản lý local repo theo kiểu git. **Yêu cầu git + git-lfs** cài sẵn.

```python
from huggingface_hub import Repository

# Clone repository về local
repo = Repository("<path_to_dummy_folder>", clone_from="<namespace>/dummy-model")

# Pull về để đồng bộ
repo.git_pull()

# Lưu model và tokenizer vào folder local
model.save_pretrained("<path_to_dummy_folder>")
tokenizer.save_pretrained("<path_to_dummy_folder>")

# Git workflow: add → commit → push
repo.git_add()
repo.git_commit("Add model and tokenizer files")
repo.git_push()
```

Các phương thức git của `Repository`:
- `git_pull()`, `git_add()`, `git_commit()`, `git_push()`, `git_tag()`

### 3.4. Cách 3: Sử dụng Git + Git-LFS trực tiếp

Đây là cách "thủ công" nhất, dùng git và git-lfs trực tiếp.

#### Bước 1: Cài đặt git-lfs:
```bash
git lfs install
```

#### Bước 2: Clone repository:
```bash
git clone https://huggingface.co/<namespace>/<your-model-id>
```

#### Bước 3: Tạo model và tokenizer:
```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

checkpoint = "camembert-base"
model = AutoModelForMaskedLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

model.save_pretrained("<path_to_dummy_folder>")
tokenizer.save_pretrained("<path_to_dummy_folder>")
```

**Các file được tạo ra** (PyTorch):
- `config.json` — Model configuration
- `pytorch_model.bin` — Model weights (~400+ MB, file lớn)
- `sentencepiece.bpe.model` — Tokenizer vocabulary
- `special_tokens_map.json`, `tokenizer_config.json`, `tokenizer.json` — Tokenizer config

**Với TensorFlow**: thay `pytorch_model.bin` bằng `tf_model.h5`.

#### Bước 4: Git workflow:
```bash
git add .
git status           # Kiểm tra các file staged
git lfs status       # Kiểm tra file nào được git-lfs track
git commit -m "First model version"
git push
```

> 💡 **Mẹo quan trọng**: Khi tạo repo từ web interface, file `.gitattributes` tự động được cấu hình để git-lfs track các file đuôi như `.bin`, `.h5`. Không cần cấu hình thêm.

---

## 4. Building a Model Card (Xây dựng Model Card)

**Model card** là file `README.md` — quan trọng không kém model và tokenizer. Nó đảm bảo:
- **Reusability** (khả năng tái sử dụng)
- **Reproducibility** (khả năng tái tạo kết quả)
- **Fairness** (tính công bằng)

### Các phần nên có trong model card:

| Phần | Nội dung |
|------|----------|
| **Model description** | Kiến trúc, version, paper gốc, tác giả, tổng quan, copyright |
| **Intended uses & limitations** | Use case phù hợp, ngôn ngữ, lĩnh vực, giới hạn đã biết |
| **How to use** | Code mẫu dùng `pipeline()`, model class, tokenizer |
| **Training data** | Dataset đã dùng, mô tả ngắn về dataset |
| **Training procedure** | Tiền xử lý, số epochs, batch size, learning rate... |
| **Variable and metrics** | Metrics dùng để đánh giá, dataset split |
| **Evaluation results** | Kết quả đánh giá, decision threshold |

### Model card metadata (YAML header):

Metadata ở đầu file README cho phép Hub phân loại mô hình:

```yaml
---
language: fr
license: mit
datasets:
- oscar
metrics:
- accuracy
---
```

Hub dùng metadata này để filter model theo task, ngôn ngữ, license, dataset, metrics.

---

## 5. Tổng kết các điểm kỹ thuật quan trọng cần nắm

### 5.1. Các cách upload model lên Hub (so sánh):

| Phương pháp | Ưu điểm | Hạn chế |
|-------------|----------|---------|
| `push_to_hub=True` trong `TrainingArguments` | Tự động hoàn toàn, sinh model card | Chỉ dùng được với `Trainer` |
| `model.push_to_hub()` / `tokenizer.push_to_hub()` | Đơn giản, trực tiếp trên object | Cần gọi riêng cho model và tokenizer |
| `upload_file()` | Không cần git/git-lfs, đơn giản | Giới hạn file < 5GB |
| `Repository` class | Quản lý local như git, nhiều tính năng | Cần git + git-lfs |
| Git + git-lfs thủ công | Kiểm soát đầy đủ | Phức tạp, dễ sai |

### 5.2. Các class `Auto*` luôn được khuyến nghị hơn class cụ thể:

```python
# ✅ Tốt (architecture-agnostic)
AutoTokenizer.from_pretrained("camembert-base")
AutoModelForMaskedLM.from_pretrained("bert-base-uncased")  # Dễ dàng đổi checkpoint

# ❌ Không nên (giới hạn vào 1 architecture)
CamembertTokenizer.from_pretrained("camembert-base")
```

### 5.3. Các câu hỏi quiz và đáp án quan trọng:

1. **Hub giới hạn loại model gì?** → **Không giới hạn** (không yêu cầu về thư viện, interface, hay lĩnh vực)
2. **Quản lý model trên Hub qua gì?** → **Git và git-lfs** (mỗi model là một Git repo, git-lfs cho file lớn)
3. **Web interface làm được gì?** → Tạo repo mới, quản lý/sửa files, upload files, xem diffs (không fork được)
4. **Model card là gì?** → Công cụ đảm bảo **reproducibility, reusability, fairness** (file Markdown, không phải Python)
5. **Object nào có `push_to_hub()`?** → Tokenizer, model configuration, model, Trainer/PushToHubCallback — **tất cả**
6. **Bước đầu khi dùng `push_to_hub()` / CLI?** → Chạy `huggingface-cli login` (terminal) hoặc `notebook_login()` (notebook)
7. **Upload model + tokenizer lên Hub?** → Gọi `push_to_hub()` trên cả model và tokenizer
8. **`Repository` class hỗ trợ thao tác git nào?** → commit, pull, push (không hỗ trợ merge)