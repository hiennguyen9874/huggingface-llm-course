# CHƯƠNG 3: FINE-TUNING MÔ HÌNH PRETRAINED

## 1. Giới thiệu

Chương này tập trung vào việc **fine-tune** (tinh chỉnh) một mô hình pretrained cho một tác vụ cụ thể. Các nội dung chính:

- Chuẩn bị dataset lớn từ Hugging Face Hub bằng thư viện `🤗 Datasets`
- Sử dụng API `Trainer` cấp cao để fine-tune với các best practices hiện đại
- Tự cài đặt training loop tùy chỉnh với các kỹ thuật tối ưu
- Sử dụng `🤗 Accelerate` để chạy distributed training
- Các kỹ thuật tối ưu hiện tại để đạt hiệu suất cao nhất

**Framework**: Chỉ sử dụng **PyTorch**.

---

## 2. Xử lý dữ liệu (Data Processing)

### 2.1. Dataset sử dụng: MRPC (Microsoft Research Paraphrase Corpus)

- 5,801 cặp câu, với nhãn cho biết chúng có phải là paraphrase (diễn đạt cùng ý) hay không
- Một phần của benchmark GLUE
- Nhãn: `0` = `not_equivalent`, `1` = `equivalent`

### 2.2. Tải dataset từ Hub

```python
from datasets import load_dataset

raw_datasets = load_dataset("glue", "mrpc")
# Kết quả: DatasetDict với train (3668), validation (408), test (1725)

raw_train_dataset = raw_datasets["train"]
raw_train_dataset[0]  # Truy cập như dictionary
# => {'idx': 0, 'label': 1, 'sentence1': ..., 'sentence2': ...}

raw_train_dataset.features  # Xem kiểu dữ liệu từng cột
# => label: ClassLabel(num_classes=2, names=['not_equivalent', 'equivalent'])
```

### 2.3. Tokenization

Đối với tác vụ phân loại cặp câu, model BERT mong đợi định dạng input:

```
[CLS] sentence1 [SEP] sentence2 [SEP]
```

Token type IDs: sentence1 = 0, sentence2 = 1. Đây là di sản từ pretraining objective **Next Sentence Prediction (NSP)** của BERT.

**Tokenize bằng `Dataset.map()` với `batched=True`** — đây là cách hiệu quả:

```python
from transformers import AutoTokenizer

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(
        example["sentence1"], example["sentence2"], truncation=True
    )

# batched=True: tokenize theo batch, nhanh hơn rất nhiều
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
```

> **Lưu ý quan trọng**: Không truyền `padding=True` trong `tokenize_function`. Thay vào đó, dùng **dynamic padding** để tránh padding toàn bộ dataset về max length.

### 2.4. Dynamic Padding — Kỹ thuật cốt lõi

Dynamic padding chỉ pad đến độ dài tối đa **trong mỗi batch**, không phải toàn bộ dataset. Điều này tiết kiệm rất nhiều tính toán khi các sequences có độ dài khác nhau lớn.

Triển khai thông qua **Data Collator**:

```python
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Test: lấy 8 mẫu đầu, bỏ cột string
samples = tokenized_datasets["train"][:8]
samples = {k: v for k, v in samples.items()
           if k not in ["idx", "sentence1", "sentence2"]}

batch = data_collator(samples)
# Kết quả: attention_mask (8, 67), input_ids (8, 67), labels (8,)
# Các mẫu được pad về max length trong batch = 67
```

---

## 3. Fine-Tuning với Trainer API

### 3.1. Các bước cơ bản

```python
from transformers import (
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)

# 1. Training arguments
training_args = TrainingArguments(
    "test-trainer",
    eval_strategy="epoch",       # Đánh giá cuối mỗi epoch
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# 2. Model (BERT với classification head cho 2 nhãn)
model = AutoModelForSequenceClassification.from_pretrained(
    checkpoint, num_labels=2
)
# Cảnh báo: pretrained head bị loại bỏ, head mới được khởi tạo ngẫu nhiên

# 3. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    processing_class=tokenizer,  # Tokenizer cho xử lý dữ liệu
    compute_metrics=compute_metrics,  # Hàm đánh giá tùy chỉnh
)

# 4. Train
trainer.train()
```

### 3.2. Hàm `compute_metrics`

```python
import numpy as np
import evaluate

def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
```

`EvalPrediction` là named tuple gồm `predictions` (logits) và `label_ids`. Ta dùng `argmax` để chuyển logits → class prediction rồi tính accuracy/F1 qua `evaluate`.

### 3.3. Các tính năng nâng cao

| Tính năng | Cách bật | Mục đích |
|---|---|---|
| **Mixed Precision** | `fp16=True` | Training nhanh hơn, tiết kiệm VRAM (dùng float16 cho forward/backward, float32 cho gradient) |
| **Gradient Accumulation** | `gradient_accumulation_steps=4` | Mô phỏng batch size lớn hơn khi VRAM hạn chế |
| **Learning Rate Scheduling** | `lr_scheduler_type="cosine"` | Chọn loại scheduler (mặc định: linear decay) |
| **Early Stopping** | `EarlyStoppingCallback(early_stopping_patience=3)` | Dừng training khi validation loss không cải thiện |
| **Push to Hub** | `push_to_hub=True` | Tự động upload model lên Hub |

---

## 4. Custom Training Loop với PyTorch + Accelerate

### 4.1. Chuẩn bị DataLoader

```python
from torch.utils.data import DataLoader

# Post-process dataset
tokenized_datasets = tokenized_datasets.remove_columns(
    ["sentence1", "sentence2", "idx"]
)
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch")  # Trả về PyTorch tensors

# DataLoader với collate_fn để dynamic padding
train_dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    batch_size=8,
    collate_fn=data_collator,
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"],
    batch_size=8,
    collate_fn=data_collator,
)
```

### 4.2. Optimizer & Scheduler

```python
from torch.optim import AdamW
from transformers import get_scheduler

# AdamW = Adam + decoupled weight decay
optimizer = AdamW(model.parameters(), lr=5e-5)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",                    # Linear decay từ max → 0
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)
```

### 4.3. Training Loop đầy đủ

```python
import torch
from tqdm.auto import tqdm

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        
        outputs = model(**batch)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        
        progress_bar.update(1)
```

### 4.4. Evaluation Loop

```python
import evaluate

metric = evaluate.load("glue", "mrpc")

model.eval()
for batch in eval_dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
    
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    metric.add_batch(predictions=predictions, references=batch["labels"])

final_metrics = metric.compute()
# => {'accuracy': 0.84..., 'f1': 0.89...}
```

> **Hai thứ luôn đi cùng evaluation**: `model.eval()` (tắt dropout, dùng running stats của batch norm) và `torch.no_grad()` (tắt gradient tracking để tiết kiệm VRAM).

### 4.5. Distributed Training với 🤗 Accelerate

Chỉ cần **3 thay đổi tối thiểu** là code chạy được trên multi-GPU/TPU:

```python
from accelerate import Accelerator

accelerator = Accelerator()

# Bọc các object cần thiết bằng accelerator.prepare()
train_dl, eval_dl, model, optimizer = accelerator.prepare(
    train_dataloader, eval_dataloader, model, optimizer
)

# Trong training loop:
# - Không cần tự move batch tới device (Accelerate tự lo)
# - Thay loss.backward() bằng:
accelerator.backward(loss)
```

Chạy distributed training:
```bash
accelerate config           # Cấu hình môi trường
accelerate launch train.py  # Chạy training phân tán
```

---

## 5. Learning Curves — Đọc hiểu đồ thị học

### 5.1. Hai loại curve chính

| Curve | Xu hướng tốt | Đặc điểm |
|---|---|---|
| **Loss curve** | Giảm dần, mượt mà | Giá trị liên tục, cải thiện ngay cả khi prediction chưa đúng |
| **Accuracy curve** | Tăng dần, có "bậc thang" (plateaus) | Rời rạc — chỉ thay đổi khi prediction vượt ngưỡng quyết định |

**Tại sao accuracy có dạng bậc thang?** Trong binary classification (ngưỡng 0.5), nếu model dự đoán 0.3 → 0.4 cho một mẫu dog (label=1), loss giảm nhưng accuracy không đổi (vẫn sai). Chỉ khi dự đoán > 0.5 thì accuracy mới nhảy.

### 5.2. Các pattern quan trọng

#### Overfitting
- **Dấu hiệu**: Training loss ↓ nhưng validation loss ↑ (hoặc plateau). Khoảng cách giữa train/val accuracy lớn.
- **Giải pháp**:
  - Early stopping (`EarlyStoppingCallback`)
  - Dropout, weight decay
  - Data augmentation
  - Giảm model capacity

```python
from transformers import EarlyStoppingCallback

trainer = Trainer(
    ...,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)
```

#### Underfitting
- **Dấu hiệu**: Cả train và val loss đều cao, plateau sớm.
- **Giải pháp**: Tăng model capacity, train lâu hơn, điều chỉnh learning rate, kiểm tra chất lượng dữ liệu.

#### Erratic Curves (dao động mạnh)
- **Dấu hiệu**: Loss/accuracy dao động lớn, không có xu hướng rõ ràng.
- **Giải pháp**: Giảm learning rate, tăng batch size, gradient clipping, kiểm tra lại preprocessing.

### 5.3. Theo dõi với Weights & Biases

```python
import wandb

wandb.init(project="transformer-fine-tuning", name="bert-mrpc")

training_args = TrainingArguments(
    ...,
    report_to="wandb",     # Gửi log tới W&B
    logging_steps=10,      # Log sau mỗi 10 steps
)
```

---

## 6. Tổng kết — Những điểm cốt lõi cần nhớ

### Pipeline Fine-Tuning hoàn chỉnh

```
Raw text → Tokenizer (map batched) → DataCollatorWithPadding (dynamic padding)
  → DataLoader → Model + Optimizer + Scheduler → Training Loop
  → Evaluation (model.eval() + torch.no_grad()) → Hub upload
```

### So sánh Trainer API vs Custom Loop

| Tiêu chí | Trainer API | Custom Loop |
|---|---|---|
| Độ phức tạp | Thấp, vài dòng code | Cao, tự viết mọi thứ |
| Linh hoạt | Giới hạn bởi API | Hoàn toàn kiểm soát |
| Distributed | Tự động (multi-GPU/TPU) | Cần Accelerate |
| Logging/Callback | Built-in | Tự cài đặt |
| Khi nào dùng | Fine-tuning tiêu chuẩn | Nghiên cứu, logic đặc biệt |

### Các tham số quan trọng trong TrainingArguments

| Tham số | Ý nghĩa |
|---|---|
| `eval_strategy` | `"epoch"`, `"steps"`, `"no"` — khi nào đánh giá |
| `learning_rate` | Tốc độ học (thường 2e-5 đến 5e-5) |
| `per_device_train_batch_size` | Batch size trên mỗi GPU |
| `gradient_accumulation_steps` | Tích lũy gradient qua N bước |
| `fp16` | Mixed precision training |
| `weight_decay` | Hệ số weight decay cho AdamW |
| `lr_scheduler_type` | `"linear"`, `"cosine"`, `"constant"` |
| `load_best_model_at_end` | Load checkpoint tốt nhất khi kết thúc |
| `push_to_hub` | Tự động upload lên Hugging Face Hub |

### Thứ tự chính xác trong training loop

```
forward pass → loss.backward() → optimizer.step() → lr_scheduler.step() → optimizer.zero_grad()
```

### Key takeaways

- **Dynamic padding**: Luôn dùng `DataCollatorWithPadding` thay vì padding cố định — tiết kiệm tính toán đáng kể.
- **`batched=True`** trong `Dataset.map()`: Tokenize nhanh hơn nhiều lần nhờ Rust tokenizer xử lý batch.
- **AdamW > Adam**: Weight decay tách rời khỏi gradient update → regularization tốt hơn.
- **`processing_class=tokenizer`**: Cho Trainer biết tokenizer nào để tự động xử lý dữ liệu.
- **Accelerate**: Chỉ cần `prepare()` + `accelerator.backward()` là code chạy được distributed.
- **Learning curves**: Công cụ debug quan trọng nhất — học cách đọc chúng để phát hiện overfitting/underfitting/learning rate sai.