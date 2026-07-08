# Chapter 8 — Debugging và yêu cầu hỗ trợ trong Hugging Face

## 1. Mục tiêu chính của chương

Chương này không dạy thêm một task NLP mới, mà dạy cách xử lý khi:

- Code bị lỗi khi load model/tokenizer/pipeline.
- Forward pass của model bị lỗi.
- Training bằng `Trainer.train()` hoặc `model.fit()` bị lỗi.
- Training chạy được nhưng model học rất tệ.
- Cần hỏi cộng đồng Hugging Face Forums.
- Cần báo bug lên GitHub issue.

Điểm quan trọng nhất:

> Debug ML không nên bắt đầu bằng đoán mò. Hãy kiểm tra pipeline từng bước: data → batch/dataloader → model forward → loss → backward → optimizer → evaluation/metrics.

---

# 2. Khi gặp Python error: đọc traceback đúng cách

Python traceback nên đọc **từ dưới lên trên**.

Dòng cuối thường chứa:

- Loại exception, ví dụ `OSError`, `AttributeError`, `ValueError`, `IndexError`.
- Message quan trọng nhất.
- Gợi ý nguyên nhân.

Ví dụ:

```text
OSError: Can't load config for 'lewtun/distillbert-base-uncased-finetuned-squad-d5716d28'
```

Cách đọc:

1. Nhìn dòng cuối để biết lỗi là gì.
2. Nếu chưa đủ, đi ngược lên trên để xem lỗi phát sinh ở hàm nào.
3. Xác định dòng code của mình gây lỗi.
4. Search nguyên văn message nếu chưa hiểu.

Nếu dùng Colab, cần mở rộng toàn bộ traceback vì Colab hay collapse stack trace.

---

# 3. Debug lỗi khi dùng `pipeline`

Ví dụ load model question-answering:

```python
from transformers import pipeline

reader = pipeline(
    "question-answering",
    model="lewtun/distillbert-base-uncased-finetuned-squad-d5716d28"
)
```

Lỗi:

```text
OSError: Can't load config for ...
```

Nguyên nhân đầu tiên: model ID sai. `distillbert` bị thừa chữ `l`; đúng là `distilbert`.

Sau khi sửa model ID vẫn lỗi:

```python
model_checkpoint = "lewtun/distilbert-base-uncased-finetuned-squad-d5716d28"
```

Kiểm tra repo trên Hub:

```python
from huggingface_hub import list_repo_files

list_repo_files(repo_id=model_checkpoint)
```

Kết quả thiếu file:

```text
config.json
```

Model Transformers cần nhiều file quan trọng:

- `config.json`: cấu hình kiến trúc model.
- `pytorch_model.bin` hoặc `model.safetensors`: weights.
- tokenizer files như `vocab.txt`, `tokenizer_config.json`, `special_tokens_map.json`.

Nếu thiếu `config.json`, `pipeline` không biết phải dựng model class nào.

Có thể lấy config từ base model:

```python
from transformers import AutoConfig

config = AutoConfig.from_pretrained("distilbert-base-uncased")
config.push_to_hub(model_checkpoint, commit_message="Add config.json")
```

Lưu ý kỹ thuật:

> Cách này chỉ an toàn nếu model fine-tune không thay đổi config gốc. Nếu người train thay đổi `num_labels`, dropout, kiến trúc... thì dùng config gốc có thể sai.

---

# 4. Debug forward pass của model

Sau khi lấy model/tokenizer từ pipeline:

```python
tokenizer = reader.tokenizer
model = reader.model
```

Code lỗi:

```python
inputs = tokenizer(question, context, add_special_tokens=True)
outputs = model(**inputs)
```

Lỗi:

```text
AttributeError: 'list' object has no attribute 'size'
```

Nguyên nhân:

- Tokenizer mặc định trả về Python `list`.
- Model PyTorch cần tensor.
- PyTorch tensor có `.size()`, list thì không.

Sai:

```python
inputs = tokenizer(question, context)
```

Đúng:

```python
inputs = tokenizer(
    question,
    context,
    add_special_tokens=True,
    return_tensors="pt"
)

outputs = model(**inputs)
```

Ví dụ đầy đủ:

```python
import torch

question = "Which frameworks can I use?"

inputs = tokenizer(
    question,
    context,
    add_special_tokens=True,
    return_tensors="pt"
)

input_ids = inputs["input_ids"][0]

outputs = model(**inputs)

answer_start = torch.argmax(outputs.start_logits)
answer_end = torch.argmax(outputs.end_logits) + 1

answer = tokenizer.convert_tokens_to_string(
    tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end])
)

print(answer)
```

Điểm cần nhớ:

| Framework | `return_tensors` |
|---|---|
| PyTorch | `"pt"` |
| TensorFlow | `"tf"` |
| NumPy | `"np"` |

---

# 5. Hỏi trên Hugging Face Forums thế nào cho hiệu quả

Một câu hỏi tốt cần có:

## 5.1 Tiêu đề rõ ràng

Không nên:

```text
Help me please!!!
```

Nên:

```text
Source of IndexError in AutoModel forward pass?
```

Hoặc:

```text
Why does my model produce an IndexError with long inputs?
```

## 5.2 Format code bằng Markdown

Dùng triple backticks:

````markdown
```python
from transformers import AutoTokenizer, AutoModel
...
```
````

Dùng inline code cho biến hoặc model name:

```markdown
I am using `distilbert-base-uncased`.
```

## 5.3 Include full traceback

Không chỉ paste dòng cuối. Full traceback giúp người khác biết lỗi xảy ra ở đâu.

## 5.4 Có reproducible example

Một ví dụ tốt phải:

- Chạy được.
- Ngắn.
- Không phụ thuộc file private.
- Có đủ import.
- Có input gây lỗi.
- Có traceback đầy đủ.

Ví dụ lỗi sequence quá dài:

```python
from transformers import AutoTokenizer, AutoModel

model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModel.from_pretrained(model_checkpoint)

text = "..."  # đoạn text rất dài

inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
```

Lỗi có thể do input dài hơn giới hạn model:

```text
Token indices sequence length is longer than the specified maximum sequence length for this model (583 > 512)
```

Với BERT/DistilBERT, max length thường là 512 tokens.

Cách xử lý:

```python
inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    max_length=512
)
```

---

# 6. Debug training pipeline với PyTorch `Trainer`

Khi `trainer.train()` lỗi, nguyên nhân có thể đến từ nhiều chỗ:

1. Dataset sai.
2. Tokenization sai.
3. Data collator/padding sai.
4. Batch không tạo được.
5. Model forward lỗi.
6. Loss lỗi.
7. Backward/optimizer lỗi.
8. Evaluation/metric lỗi.

Không nên debug trực tiếp từ `trainer.train()` vì nó gom quá nhiều thứ. Hãy tách pipeline.

---

## 6.1 Kiểm tra dataset đầu vào

Ví dụ script sai:

```python
tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

trainer = Trainer(
    model,
    args,
    train_dataset=raw_datasets["train"],
    eval_dataset=raw_datasets["validation_matched"],
)
```

Lỗi:

```text
ValueError: You have to specify either input_ids or inputs_embeds
```

Nguyên nhân:

- Đã tokenize nhưng lại truyền `raw_datasets` vào Trainer.
- Raw dataset chỉ có text, chưa có `input_ids`.
- Trainer loại bỏ cột không nằm trong signature của model.
- Model không nhận được `input_ids`.

Đúng:

```python
trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation_matched"],
)
```

Luôn kiểm tra:

```python
trainer.train_dataset[0]
```

Kiểm tra decoded input:

```python
tokenizer.decode(trainer.train_dataset[0]["input_ids"])
```

Kiểm tra keys:

```python
trainer.train_dataset[0].keys()
```

Kiểm tra attention mask:

```python
len(trainer.train_dataset[0]["attention_mask"]) == len(
    trainer.train_dataset[0]["input_ids"]
)
```

Kiểm tra label mapping:

```python
trainer.train_dataset.features["label"].names
```

Ví dụ MNLI có 3 nhãn:

```python
['entailment', 'neutral', 'contradiction']
```

---

## 6.2 Kiểm tra dataloader và data collator

Tạo thử batch đầu tiên:

```python
for batch in trainer.get_train_dataloader():
    break
```

Nếu lỗi:

```text
ValueError: expected sequence of length 43 at dim 1 (got 37)
```

Thường do các sequence có độ dài khác nhau nhưng không được padding.

Sai: dùng default collator.

Đúng: dùng `DataCollatorWithPadding`.

```python
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation_matched"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)
```

Nếu muốn test collator thủ công:

```python
data_collator = trainer.get_train_dataloader().collate_fn
actual_train_set = trainer._remove_unused_columns(trainer.train_dataset)

batch = data_collator([actual_train_set[i] for i in range(4)])
```

---

## 6.3 Kiểm tra model forward

Sau khi có batch:

```python
outputs = trainer.model.cpu()(**batch)
```

Nếu gặp CUDA error khó hiểu, hãy chạy trên CPU.

Ví dụ CUDA lỗi:

```text
RuntimeError: CUDA error: CUBLAS_STATUS_ALLOC_FAILED
```

Chạy CPU cho ra lỗi thật:

```text
IndexError: Target 2 is out of bounds.
```

Nguyên nhân:

- Dataset có label `0, 1, 2`.
- Model được tạo mặc định với `num_labels=2`.
- Label `2` vượt ngoài số class của model.

Sai:

```python
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
```

Đúng:

```python
model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels=3
)
```

---

## 6.4 Kiểm tra backward và optimizer

Sau khi forward chạy được:

```python
loss = outputs.loss
loss.backward()
```

Tạo optimizer:

```python
trainer.create_optimizer()
trainer.optimizer.step()
trainer.optimizer.zero_grad()
```

Nếu optimizer custom lỗi, debug ở CPU trước.

---

## 6.5 CUDA error và CUDA OOM

Có hai nhóm:

### CUDA error không rõ nguyên nhân

Do GPU chạy bất đồng bộ, traceback thường chỉ ra sai vị trí.

Cách debug tốt nhất:

```python
outputs = trainer.model.cpu()(**batch)
```

CPU thường cho lỗi rõ hơn.

### CUDA out of memory

Lỗi kiểu:

```text
RuntimeError: CUDA out of memory
```

Cách xử lý:

- Restart kernel.
- Giảm batch size.
- Đảm bảo không có nhiều model nằm trên GPU cùng lúc.
- Dùng model nhỏ hơn.
- Dùng gradient accumulation nếu cần giữ effective batch size.
- Dùng mixed precision nếu phù hợp.

---

## 6.6 Kiểm tra evaluation và metrics

Nên chạy evaluation trước khi train lâu:

```python
trainer.evaluate()
```

Ví dụ lỗi:

```text
TypeError: only size-1 arrays can be converted to Python scalars
```

Nguyên nhân trong `compute_metrics()`:

```python
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    return metric.compute(predictions=predictions, references=labels)
```

`predictions` đang là logits shape `(batch_size, num_labels)`, không phải class IDs.

Đúng:

```python
import numpy as np

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)
```

---

## 6.7 Script PyTorch hoàn chỉnh đã sửa

```python
import numpy as np
from datasets import load_dataset
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

raw_datasets = load_dataset("glue", "mnli")

model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

def preprocess_function(examples):
    return tokenizer(
        examples["premise"],
        examples["hypothesis"],
        truncation=True,
    )

tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels=3,
)

args = TrainingArguments(
    "distilbert-finetuned-mnli",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
)

metric = evaluate.load("glue", "mnli")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation_matched"],
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()
```

---

# 7. Debug training pipeline với TensorFlow/Keras

TensorFlow version dùng:

```python
model.fit(train_dataset)
```

Các lỗi chính trong chương:

## 7.1 Labels đặt sai vị trí

Ví dụ:

```python
train_dataset = tokenized_datasets["train"].to_tf_dataset(
    columns=["input_ids", "labels"],
    batch_size=16,
    shuffle=True,
)
```

Sau đó:

```python
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")
model.fit(train_dataset)
```

Lỗi:

```text
ValueError: No gradients provided for any variable
```

Nguyên nhân:

- Keras loss cần labels tách riêng.
- Nhưng labels đang nằm trong input dictionary.
- Transformers model có thể tự tính loss nội bộ nếu labels nằm trong input.
- Nhưng khi truyền `loss=...`, Keras cố tự tính loss và không nhận đúng labels.

Cách đơn giản:

```python
model.compile(optimizer="adam")
```

Khi không truyền loss, Transformers model dùng internal loss.

---

## 7.2 NaN loss vì sai `num_labels`

Model mặc định có 2 labels, nhưng MNLI có 3 labels.

Sai:

```python
model = TFAutoModelForSequenceClassification.from_pretrained(model_checkpoint)
```

Đúng:

```python
model = TFAutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels=3
)
```

Kỹ thuật debug NaN:

```python
for batch in train_dataset:
    break

outputs = model(batch)
print(outputs.loss)
print(outputs.logits)
```

Nếu loss NaN ở các sample có label 2, kiểm tra:

```python
model.config.num_labels
```

---

## 7.3 Learning rate mặc định của Adam quá cao

Keras:

```python
model.compile(optimizer="adam")
```

sẽ dùng Adam mặc định với learning rate `1e-3`.

Với Transformer, `1e-3` thường quá cao.

Nên dùng khoảng:

```text
1e-5 đến 5e-5
```

Ví dụ:

```python
from tensorflow.keras.optimizers import Adam

model = TFAutoModelForSequenceClassification.from_pretrained(
    model_checkpoint,
    num_labels=3,
)

model.compile(optimizer=Adam(5e-5))
model.fit(train_dataset)
```

---

# 8. Silent errors: training chạy nhưng model học tệ

Đây là phần khó nhất trong ML. Chương gợi ý các bước sau.

## 8.1 Check data lại lần nữa

Hỏi các câu:

- Decoded input có hợp lý không?
- Label có đúng không?
- Có class nào áp đảo không?
- Nếu model đoán random thì metric/loss nên khoảng bao nhiêu?
- Nếu model luôn đoán class phổ biến nhất thì metric bao nhiêu?
- Loss ban đầu có hợp lý không?

Decode input:

```python
tokenizer.decode(trainer.train_dataset[0]["input_ids"])
```

Với TensorFlow:

```python
input_ids = batch["input_ids"].numpy()
tokenizer.decode(input_ids[0])
```

## 8.2 Overfit trên một batch

Đây là test cực kỳ quan trọng.

Ý tưởng:

> Nếu model không thể memorize một batch nhỏ, thì pipeline/data/loss/model framing có vấn đề.

PyTorch:

```python
for batch in trainer.get_train_dataloader():
    break

batch = {k: v.to(device) for k, v in batch.items()}

trainer.create_optimizer()

for _ in range(20):
    outputs = trainer.model(**batch)
    loss = outputs.loss
    loss.backward()
    trainer.optimizer.step()
    trainer.optimizer.zero_grad()
```

Sau đó kiểm tra metric trên chính batch đó. Kỳ vọng gần perfect.

TensorFlow:

```python
for batch in train_dataset:
    break

model.fit(batch, epochs=20)
```

Lưu ý:

> Sau test overfit một batch, nên tạo lại model/trainer từ đầu. Model đã bị ép memorize batch nhỏ, không còn phù hợp để train thật.

---

## 8.3 Đừng hyperparameter tuning quá sớm

Không nên chạy hàng trăm experiment khi chưa có baseline.

Thứ tự tốt hơn:

1. Data đúng.
2. Batch đúng.
3. Forward/loss đúng.
4. Metric đúng.
5. Overfit một batch được.
6. Có baseline hợp lý.
7. Sau đó mới tune learning rate, batch size, weight decay...

---

# 9. Cách viết GitHub issue tốt

Chỉ mở GitHub issue khi khá chắc đây là bug của thư viện, không phải lỗi code của mình. Nếu chưa chắc, hỏi forum trước.

Một issue tốt cần:

## 9.1 Minimal reproducible example

Nghĩa là:

- Ngắn.
- Tự chạy được.
- Không cần private data/file.
- Có code đầy đủ.
- Reproduce đúng lỗi.

Không tốt:

- Screenshot traceback.
- Notebook dài hàng trăm cell.
- Code phụ thuộc data nội bộ.

Tốt:

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("...")
model = AutoModel.from_pretrained("...")

inputs = tokenizer("...", return_tensors="pt")
outputs = model(**inputs)
```

## 9.2 Environment info

Dùng:

```bash
transformers-cli env
```

Thông tin này giúp maintainers biết:

- Transformers version.
- Python version.
- PyTorch/TensorFlow version.
- OS.
- GPU/CPU.
- Distributed setup hay không.

## 9.3 Full traceback

Paste nguyên traceback trong code block:

````markdown
```text
Traceback ...
```
````

## 9.4 Expected behavior

Nói rõ bạn mong đợi gì:

```text
I expected the model to return logits with shape [batch_size, num_labels], but it raises IndexError.
```

## 9.5 Tag người khác vừa phải

Không tag bừa. Nếu cần, tag tối đa vài người liên quan trực tiếp.

---

# 10. Checklist debug kỹ thuật quan trọng

## Khi lỗi ở inference/pipeline

- [ ] Model ID đúng chưa?
- [ ] Repo có `config.json` không?
- [ ] Repo có weights không?
- [ ] Tokenizer files đủ không?
- [ ] Có đang dùng đúng `revision` không?
- [ ] Tokenizer trả tensor chưa? `return_tensors="pt"` hoặc `"tf"`?
- [ ] Input có vượt max length không?

## Khi lỗi ở `Trainer.train()`

- [ ] `train_dataset` là dataset đã tokenize chưa?
- [ ] Có `input_ids`, `attention_mask`, `labels` không?
- [ ] Decode thử `input_ids`.
- [ ] Label mapping đúng không?
- [ ] `num_labels` của model đúng không?
- [ ] Data collator có padding đúng không?
- [ ] Tạo được một batch không?
- [ ] Batch forward qua model được không?
- [ ] Loss backward được không?
- [ ] Optimizer step được không?
- [ ] `compute_metrics` có convert logits sang prediction không?
- [ ] `trainer.evaluate()` chạy được trước khi train lâu không?

## Khi lỗi CUDA

- [ ] Nếu không phải OOM, chạy lại trên CPU để lấy lỗi rõ hơn.
- [ ] Nếu OOM, giảm batch size.
- [ ] Restart kernel sau CUDA crash.
- [ ] Kiểm tra có model/process khác đang giữ GPU không.

## Khi model học tệ

- [ ] Check decoded data.
- [ ] Check labels.
- [ ] Check imbalance.
- [ ] Check metric/loss implementation.
- [ ] Overfit một batch.
- [ ] Check learning rate.
- [ ] Check weight decay.
- [ ] Tạo baseline trước khi tuning lớn.

---

# 11. Những bài học quan trọng nhất

1. **Đọc traceback từ dưới lên.**
2. **Đừng debug `trainer.train()` như một khối đen. Tách từng bước.**
3. **Luôn kiểm tra data đầu tiên.**
4. **Decode input để chắc model thấy đúng dữ liệu.**
5. **Batching/padding là nguồn lỗi rất phổ biến.**
6. **CUDA error thường gây nhiễu; debug bằng CPU.**
7. **OOM thì giảm memory usage, thường là batch size.**
8. **Classification phải set đúng `num_labels`.**
9. **Metrics thường cần `argmax(logits)` trước khi compute.**
10. **Overfit một batch là test rất mạnh để xác nhận model có thể học.**
11. **Khi hỏi forum hoặc báo bug, đưa reproducible example và full traceback.**