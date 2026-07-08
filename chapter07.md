# Chapter 7 — Các tác vụ NLP chính với Transformers

Chương này ghép các kiến thức trước đó:

- `Trainer` / `Seq2SeqTrainer` hoặc Keras.
- `datasets.Dataset.map()`, `filter()`, `train_test_split()`.
- tokenizer nhanh `FastTokenizer`, `word_ids()`, `offset_mapping`.
- data collator phù hợp từng tác vụ.
- metric như `seqeval`, `sacrebleu`, `rouge`, `squad`.
- push model lên Hugging Face Hub.
- custom training loop với 🤗 Accelerate.

Các tác vụ chính:

1. Token classification.
2. Masked language modeling.
3. Translation.
4. Summarization.
5. Causal language modeling từ đầu.
6. Question answering.
7. Tổng kết liên hệ với LLMs.

---

# 1. Token classification

## Bản chất

Token classification là bài toán gán nhãn cho từng token/word trong câu.

Ví dụ:

- **NER**: nhận diện thực thể như người, tổ chức, địa điểm.
- **POS tagging**: gán từ loại.
- **Chunking**: xác định cụm từ.

Ví dụ nhãn NER theo chuẩn BIO:

```text
EU     rejects German call ...
B-ORG  O       B-MISC O
```

Ý nghĩa:

- `O`: không phải entity.
- `B-PER`, `I-PER`: người.
- `B-ORG`, `I-ORG`: tổ chức.
- `B-LOC`, `I-LOC`: địa điểm.
- `B-MISC`, `I-MISC`: thực thể khác.

## Dataset dùng trong chương

Dataset: `conll2003`.

```python
from datasets import load_dataset

raw_datasets = load_dataset("conll2003")
```

Các cột chính:

```text
tokens
ner_tags
pos_tags
chunk_tags
```

Lấy tên nhãn:

```python
ner_feature = raw_datasets["train"].features["ner_tags"]
label_names = ner_feature.feature.names
```

## Vấn đề kỹ thuật quan trọng: align label với subword token

Input ban đầu đã được tách thành word:

```python
['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.']
```

Nhưng tokenizer của BERT có thể tách word thành subword:

```python
['[CLS]', 'EU', 'rejects', 'German', ..., 'la', '##mb', '.', '[SEP]']
```

Vấn đề:

- Số word label là 9.
- Số token sau tokenizer là 12.
- Cần align label word-level sang token-level.

Dùng fast tokenizer:

```python
inputs = tokenizer(tokens, is_split_into_words=True)
word_ids = inputs.word_ids()
```

Ví dụ:

```python
[None, 0, 1, 2, 3, 4, 5, 6, 7, 7, 8, None]
```

- `None`: special token như `[CLS]`, `[SEP]`.
- `7, 7`: word thứ 7 bị tách thành hai subword.

## Hàm align label

```python
def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None

    for word_id in word_ids:
        if word_id != current_word:
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)

        elif word_id is None:
            new_labels.append(-100)

        else:
            label = labels[word_id]
            if label % 2 == 1:  # B-XXX -> I-XXX
                label += 1
            new_labels.append(label)

    return new_labels
```

Điểm cần nhớ:

- `-100` được PyTorch loss mặc định bỏ qua.
- Special tokens phải là `-100`.
- Subword tiếp theo của một word thường đổi `B-XXX` thành `I-XXX`.
- Một biến thể khác: chỉ label subword đầu, các subword còn lại gán `-100` để word dài không ảnh hưởng loss quá nhiều.

## Preprocess toàn dataset

```python
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
    )

    all_labels = examples["ner_tags"]
    new_labels = []

    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs
```

```python
tokenized_datasets = raw_datasets.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
)
```

## Data collator

Không dùng `DataCollatorWithPadding` vì label cũng cần padding.

Dùng:

```python
from transformers import DataCollatorForTokenClassification

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
```

Nó sẽ:

- Pad input.
- Pad label cùng độ dài.
- Label padding dùng `-100`.

## Model

```python
from transformers import AutoModelForTokenClassification

id2label = {i: label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}

model = AutoModelForTokenClassification.from_pretrained(
    "bert-base-cased",
    id2label=id2label,
    label2id=label2id,
)
```

Cần kiểm tra:

```python
model.config.num_labels
```

Nếu sai số nhãn, lỗi CUDA kiểu `device-side assert triggered` rất dễ xảy ra.

## Metric

Dùng `seqeval`.

```python
import evaluate
metric = evaluate.load("seqeval")
```

`seqeval` cần label dạng string, không phải int.

```python
import numpy as np

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)

    true_labels = [
        [label_names[l] for l in label if l != -100]
        for label in labels
    ]

    true_predictions = [
        [label_names[p] for p, l in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    all_metrics = metric.compute(
        predictions=true_predictions,
        references=true_labels,
    )

    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }
```

## Inference

```python
from transformers import pipeline

token_classifier = pipeline(
    "token-classification",
    model="huggingface-course/bert-finetuned-ner",
    aggregation_strategy="simple",
)

token_classifier("My name is Sylvain and I work at Hugging Face in Brooklyn.")
```

`aggregation_strategy="simple"` gom các subword/entity liền nhau thành entity hoàn chỉnh.

---

# 2. Masked Language Modeling — MLM

## Mục tiêu

MLM là kiểu training của BERT:

- Random mask một số token.
- Model dự đoán token gốc tại vị trí bị mask.

Ví dụ:

```text
This is a great [MASK].
```

Sau fine-tune trên IMDb, model có thể dự đoán:

```text
This is a great movie.
This is a great film.
```

## Domain adaptation

Nếu domain dữ liệu khác pretraining corpus, ví dụ:

- văn bản pháp lý,
- y khoa,
- khoa học,
- review phim,
- code,

thì có thể fine-tune language model trên dữ liệu domain đó trước, sau đó fine-tune task-specific head.

Đây gọi là **domain adaptation**.

## Model dùng trong chương

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

model_checkpoint = "distilbert-base-uncased"
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
```

DistilBERT có khoảng 67M parameters, nhỏ hơn BERT-base 110M.

## Dataset

Dataset: IMDb.

```python
from datasets import load_dataset

imdb_dataset = load_dataset("imdb")
```

Splits:

```text
train: 25000
test: 25000
unsupervised: 50000
```

MLM không cần label sentiment.

## Tokenize không truncation

Quan trọng: với language modeling, không nên truncate từng example vì mất dữ liệu.

```python
def tokenize_function(examples):
    result = tokenizer(examples["text"])

    if tokenizer.is_fast:
        result["word_ids"] = [
            result.word_ids(i)
            for i in range(len(result["input_ids"]))
        ]

    return result
```

```python
tokenized_datasets = imdb_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text", "label"],
)
```

## Concatenate rồi chunk

Với LM, ta thường:

1. Tokenize từng văn bản.
2. Nối tất cả token lại.
3. Cắt thành chunk cố định.

```python
chunk_size = 128
```

```python
def group_texts(examples):
    concatenated_examples = {
        k: sum(examples[k], [])
        for k in examples.keys()
    }

    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // chunk_size) * chunk_size

    result = {
        k: [
            t[i : i + chunk_size]
            for i in range(0, total_length, chunk_size)
        ]
        for k, t in concatenated_examples.items()
    }

    result["labels"] = result["input_ids"].copy()
    return result
```

```python
lm_datasets = tokenized_datasets.map(group_texts, batched=True)
```

Điểm cần nhớ:

- `labels` ban đầu giống `input_ids`.
- Masking sẽ được data collator làm động trong mỗi batch.
- Chunk cuối nhỏ hơn `chunk_size` bị bỏ.

## Data collator cho MLM

```python
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm_probability=0.15,
)
```

Nó sẽ:

- Random mask 15% token.
- Set labels là token gốc ở vị trí cần dự đoán.
- Các vị trí không cần dự đoán gán `-100`.

## Whole word masking

Mặc định có thể mask từng subword riêng lẻ.

Whole word masking mask toàn bộ subword thuộc cùng một word.

Ý tưởng:

```python
mapping[word_index] = [token_indices]
```

Sau đó random chọn word, mask toàn bộ token indices của word đó.

Điểm kỹ thuật:

- Cần `word_ids`.
- Cần giữ cột `word_ids`.
- Nếu dùng `Trainer`, phải đặt `remove_unused_columns=False` nếu collator cần `word_ids`.

## Perplexity

Perplexity thường dùng đánh giá language model:

```python
perplexity = exp(cross_entropy_loss)
```

- Perplexity càng thấp càng tốt.
- Pretrained DistilBERT trên IMDb ban đầu khoảng 21.75.
- Sau fine-tune giảm xuống khoảng 11.32.

```python
import math

eval_results = trainer.evaluate()
print(math.exp(eval_results["eval_loss"]))
```

## Inference

```python
from transformers import pipeline

mask_filler = pipeline(
    "fill-mask",
    model="huggingface-course/distilbert-base-uncased-finetuned-imdb",
)

mask_filler("This is a great [MASK].")
```

---

# 3. Translation

## Bản chất

Translation là **sequence-to-sequence task**:

```text
source sequence -> target sequence
```

Cùng nhóm với:

- summarization,
- style transfer,
- generative QA.

## Dataset

Dataset: KDE4 English-French.

```python
from datasets import load_dataset

raw_datasets = load_dataset("kde4", lang1="en", lang2="fr")
```

Chỉ có split `train`, nên tự chia:

```python
split_datasets = raw_datasets["train"].train_test_split(
    train_size=0.9,
    seed=20,
)
split_datasets["validation"] = split_datasets.pop("test")
```

Một example:

```python
{
  "en": "Default to expanded threads",
  "fr": "Par défaut, développer les fils de discussion"
}
```

## Model

```python
model_checkpoint = "Helsinki-NLP/opus-mt-en-fr"
```

Tokenizer:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
```

Model:

```python
from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
```

## Preprocess input và target

Quan trọng: target phải tokenized đúng chế độ target.

```python
inputs = tokenizer(en_sentence, text_target=fr_sentence)
```

Nếu tokenize target như input thường, Marian có thể dùng tokenizer source cho tiếng Pháp, dẫn đến tokenization xấu hơn.

Preprocess:

```python
max_length = 128

def preprocess_function(examples):
    inputs = [ex["en"] for ex in examples["translation"]]
    targets = [ex["fr"] for ex in examples["translation"]]

    model_inputs = tokenizer(
        inputs,
        text_target=targets,
        max_length=max_length,
        truncation=True,
    )

    return model_inputs
```

```python
tokenized_datasets = split_datasets.map(
    preprocess_function,
    batched=True,
    remove_columns=split_datasets["train"].column_names,
)
```

## Data collator

Dùng:

```python
from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
```

Nó làm:

- Dynamic padding input.
- Dynamic padding labels.
- Label padding dùng `-100`.
- Tạo `decoder_input_ids` bằng cách shift labels sang phải.

Điểm quan trọng:

- Seq2Seq decoder training cần input là target tokens đã shift right.
- Padding labels bằng `pad_token_id` sẽ làm loss học cả padding, sai.
- Vì vậy label padding phải là `-100`.

## Metric: SacreBLEU

BLEU đo độ giống giữa translation và reference.

Dùng SacreBLEU vì chuẩn hóa tokenization.

```python
import evaluate

metric = evaluate.load("sacrebleu")
```

Format:

```python
predictions = ["..."]
references = [["..."]]
```

Compute metrics:

```python
import numpy as np

def compute_metrics(eval_preds):
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(
        preds,
        skip_special_tokens=True,
    )

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_labels = tokenizer.batch_decode(
        labels,
        skip_special_tokens=True,
    )

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    result = metric.compute(
        predictions=decoded_preds,
        references=decoded_labels,
    )

    return {"bleu": result["score"]}
```

## Vì sao dùng `Seq2SeqTrainer`

Với translation, evaluation nên dùng `model.generate()`, không chỉ argmax logits.

```python
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
```

Quan trọng:

```python
predict_with_generate=True
```

Vì inference thật sự là decoder generate từng token.

## Inference

```python
from transformers import pipeline

translator = pipeline(
    "translation",
    model="huggingface-course/marian-finetuned-kde4-en-to-fr",
)

translator("Default to expanded threads")
```

Sau fine-tune, model học style/domain KDE, ví dụ dịch “threads” thành “fils de discussion” thay vì giữ “threads”.

---

# 4. Summarization

## Bản chất

Summarization cũng là sequence-to-sequence:

```text
long document -> short summary
```

Chương này fine-tune mT5 để tóm tắt review Amazon bằng tiếng Anh và Tây Ban Nha.

## Dataset

Dataset: `amazon_reviews_multi`.

```python
from datasets import load_dataset

spanish_dataset = load_dataset("amazon_reviews_multi", "es")
english_dataset = load_dataset("amazon_reviews_multi", "en")
```

Cột quan trọng:

```text
review_body
review_title
language
product_category
```

Dùng:

- `review_body` làm input.
- `review_title` làm summary target.

## Lọc domain book

```python
def filter_books(example):
    return (
        example["product_category"] == "book"
        or example["product_category"] == "digital_ebook_purchase"
    )
```

```python
spanish_books = spanish_dataset.filter(filter_books)
english_books = english_dataset.filter(filter_books)
```

Gộp tiếng Anh + Tây Ban Nha:

```python
from datasets import concatenate_datasets, DatasetDict

books_dataset = DatasetDict()

for split in english_books.keys():
    books_dataset[split] = concatenate_datasets(
        [english_books[split], spanish_books[split]]
    )
    books_dataset[split] = books_dataset[split].shuffle(seed=42)
```

Lọc title quá ngắn:

```python
books_dataset = books_dataset.filter(
    lambda x: len(x["review_title"].split()) > 2
)
```

Lý do:

- Title 1-2 từ sẽ làm model học sinh summary quá ngắn và kém hữu ích.

## Model

Dùng mT5:

```python
model_checkpoint = "google/mt5-small"
```

Tokenizer:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
```

Model:

```python
from transformers import AutoModelForSeq2SeqLM

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
```

mT5 dùng SentencePiece, tốt cho multilingual.

## Preprocess

```python
max_input_length = 512
max_target_length = 30

def preprocess_function(examples):
    model_inputs = tokenizer(
        examples["review_body"],
        max_length=max_input_length,
        truncation=True,
    )

    labels = tokenizer(
        examples["review_title"],
        max_length=max_target_length,
        truncation=True,
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
```

```python
tokenized_datasets = books_dataset.map(
    preprocess_function,
    batched=True,
)
```

## Metric: ROUGE

ROUGE đo overlap giữa generated summary và reference summary.

Các loại:

- `rouge1`: overlap unigram.
- `rouge2`: overlap bigram.
- `rougeL`: longest common subsequence.
- `rougeLsum`: cho multi-sentence summary.

Cài:

```python
import evaluate

rouge_score = evaluate.load("rouge")
```

ROUGE tính precision, recall, F1. Thường báo F1.

## Baseline Lead-3

Baseline phổ biến: lấy 3 câu đầu làm summary.

```python
from nltk.tokenize import sent_tokenize

def three_sentence_summary(text):
    return "\n".join(sent_tokenize(text)[:3])
```

Với review title ngắn, lead-3 thường quá dài, ROUGE2 thấp.

## Compute metrics cho summarization

```python
import numpy as np
from nltk.tokenize import sent_tokenize

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    decoded_preds = tokenizer.batch_decode(
        predictions,
        skip_special_tokens=True,
    )

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    decoded_labels = tokenizer.batch_decode(
        labels,
        skip_special_tokens=True,
    )

    decoded_preds = [
        "\n".join(sent_tokenize(pred.strip()))
        for pred in decoded_preds
    ]

    decoded_labels = [
        "\n".join(sent_tokenize(label.strip()))
        for label in decoded_labels
    ]

    result = rouge_score.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True,
    )

    result = {
        key: value.mid.fmeasure * 100
        for key, value in result.items()
    }

    return {k: round(v, 4) for k, v in result.items()}
```

## Training

Dùng `Seq2SeqTrainer`.

```python
from transformers import Seq2SeqTrainingArguments

args = Seq2SeqTrainingArguments(
    output_dir="mt5-small-finetuned-amazon-en-es",
    evaluation_strategy="epoch",
    learning_rate=5.6e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=8,
    predict_with_generate=True,
    logging_steps=logging_steps,
    push_to_hub=True,
)
```

Collator:

```python
from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
```

Trainer:

```python
from transformers import Seq2SeqTrainer

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)
```

## Inference

```python
from transformers import pipeline

summarizer = pipeline(
    "summarization",
    model="huggingface-course/mt5-small-finetuned-amazon-en-es",
)

summarizer(review_text)
```

Model có thể tóm tắt cả tiếng Anh và Tây Ban Nha.

---

# 5. Causal Language Modeling từ đầu

## Mục tiêu

Train GPT-like model từ đầu cho code completion.

Khác với fine-tune:

- Không dùng pretrained weights.
- Khởi tạo model mới từ config.
- Cần nhiều data và compute hơn.

Dataset dùng là code Python từ GitHub, lọc các thư viện data science:

```python
filters = ["pandas", "sklearn", "matplotlib", "seaborn"]
```

## Dataset

Full dataset gốc: `transformersbook/codeparrot`.

Bản đã lọc sẵn:

```python
from datasets import load_dataset, DatasetDict

ds_train = load_dataset(
    "huggingface-course/codeparrot-ds-train",
    split="train",
)

ds_valid = load_dataset(
    "huggingface-course/codeparrot-ds-valid",
    split="validation",
)

raw_datasets = DatasetDict({
    "train": ds_train,
    "valid": ds_valid,
})
```

Cột chính:

```text
content
repo_name
path
license
```

## Tokenization thành chunk cố định

Context length nhỏ để autocomplete ngắn:

```python
context_length = 128
```

Tokenizer đã train ở chapter 6:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "huggingface-course/code-search-net-tokenizer"
)
```

Tokenize với overflowing:

```python
def tokenize(element):
    outputs = tokenizer(
        element["content"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )

    input_batch = []

    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length:
            input_batch.append(input_ids)

    return {"input_ids": input_batch}
```

```python
tokenized_datasets = raw_datasets.map(
    tokenize,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
)
```

Điểm quan trọng:

- Một file code dài sinh nhiều chunks.
- Chunk ngắn hơn `context_length` bị bỏ.
- Với corpus nhỏ hoặc context lớn, nên nối documents bằng `eos_token_id` rồi chunk để không bỏ nhiều dữ liệu.

## Khởi tạo GPT-2 model mới

```python
from transformers import AutoConfig, GPT2LMHeadModel

config = AutoConfig.from_pretrained(
    "gpt2",
    vocab_size=len(tokenizer),
    n_ctx=context_length,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

model = GPT2LMHeadModel(config)
```

Không dùng:

```python
from_pretrained(...)
```

vì đang train từ scratch.

Model khoảng 124M parameters.

## Data collator cho causal LM

```python
from transformers import DataCollatorForLanguageModeling

tokenizer.pad_token = tokenizer.eos_token

data_collator = DataCollatorForLanguageModeling(
    tokenizer,
    mlm=False,
)
```

Quan trọng:

- `mlm=False` nghĩa là causal LM.
- Labels là input IDs copy.
- Việc shift labels để predict next token xảy ra bên trong model.

## TrainingArguments

```python
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="codeparrot-ds",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    evaluation_strategy="steps",
    eval_steps=5_000,
    logging_steps=5_000,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    weight_decay=0.1,
    warmup_steps=1_000,
    lr_scheduler_type="cosine",
    learning_rate=5e-4,
    save_steps=5_000,
    fp16=True,
    push_to_hub=True,
)
```

Effective batch size:

```text
per_device_train_batch_size * gradient_accumulation_steps
```

Ví dụ:

```text
32 * 8 = 256
```

## Inference code generation

```python
from transformers import pipeline
import torch

device = 0 if torch.cuda.is_available() else -1

pipe = pipeline(
    "text-generation",
    model="huggingface-course/codeparrot-ds",
    device=device,
)

prompt = """
# create some data
x = np.random.randn(100)
y = np.random.randn(100)

# create scatter plot with x, y
"""

print(pipe(prompt, num_return_sequences=1)[0]["generated_text"])
```

Model có thể sinh:

```python
plt.scatter(x, y)
```

## Custom loss với Accelerate

Chương này còn minh họa custom loss ưu tiên sample có keyword quan trọng:

```python
["plt", "pd", "sk", "fit", "predict"]
```

Ý tưởng:

- Đếm keyword tokens trong input.
- Sample càng nhiều keyword càng có weight cao hơn.
- Tính weighted loss.

```python
def keytoken_weighted_loss(inputs, logits, keytoken_ids, alpha=1.0):
    shift_labels = inputs[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()

    loss_fct = CrossEntropyLoss(reduce=False)

    loss = loss_fct(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )

    loss_per_sample = loss.view(
        shift_logits.size(0),
        shift_logits.size(1),
    ).mean(axis=1)

    weights = torch.stack([
        (inputs == kt).float()
        for kt in keytoken_ids
    ]).sum(axis=[0, 2])

    weights = alpha * (1.0 + weights)

    weighted_loss = (loss_per_sample * weights).mean()
    return weighted_loss
```

Đây là ví dụ tốt cho trường hợp `Trainer` không đủ linh hoạt, cần custom training loop.

---

# 6. Question Answering — Extractive QA

## Bản chất

Extractive QA:

```text
question + context -> answer span trong context
```

Model không generate câu mới, mà dự đoán:

- start token index,
- end token index.

Ví dụ:

```text
Question: Which libraries back Transformers?
Context: ... Jax, PyTorch and TensorFlow ...
Answer: Jax, PyTorch and TensorFlow
```

## Dataset

Dataset: SQuAD.

```python
from datasets import load_dataset

raw_datasets = load_dataset("squad")
```

Cột:

```text
id
title
context
question
answers
```

`answers` có dạng:

```python
{
    "text": ["Saint Bernadette Soubirous"],
    "answer_start": [515]
}
```

Trong train thường có 1 answer. Validation có thể có nhiều answer hợp lệ.

## Tokenization format

Với BERT:

```text
[CLS] question [SEP] context [SEP]
```

```python
inputs = tokenizer(question, context)
```

## Vấn đề kỹ thuật lớn: context dài

Context có thể dài hơn max length.

Dùng sliding window:

```python
max_length = 384
stride = 128
```

Tokenizer:

```python
inputs = tokenizer(
    questions,
    contexts,
    max_length=max_length,
    truncation="only_second",
    stride=stride,
    return_overflowing_tokens=True,
    return_offsets_mapping=True,
    padding="max_length",
)
```

Ý nghĩa:

- `truncation="only_second"`: chỉ truncate context.
- `stride`: overlap giữa các chunks context.
- `return_overflowing_tokens=True`: một example có thể sinh nhiều features.
- `return_offsets_mapping=True`: map token về character span trong context.
- `overflow_to_sample_mapping`: feature nào thuộc example gốc nào.

## Tạo label start/end cho training

Cần chuyển answer character span sang token span.

Với mỗi feature:

1. Tìm example gốc.
2. Lấy `answer_start`.
3. Tính `answer_end`.
4. Xác định token range thuộc context.
5. Nếu answer không nằm trọn trong chunk, label là `(0, 0)` tức `[CLS]`.
6. Nếu có, tìm token start/end.

```python
def preprocess_training_examples(examples):
    questions = [q.strip() for q in examples["question"]]

    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]

    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]

        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])

        sequence_ids = inputs.sequence_ids(i)

        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx

        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions

    return inputs
```

Apply:

```python
train_dataset = raw_datasets["train"].map(
    preprocess_training_examples,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
)
```

## Preprocess validation

Validation không cần label start/end, nhưng cần giữ:

- `offset_mapping`,
- `example_id`.

Đồng thời set offset của question/special tokens thành `None`, chỉ giữ offset của context.

```python
def preprocess_validation_examples(examples):
    questions = [q.strip() for q in examples["question"]]

    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]

        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None
            for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs
```

## Post-processing QA prediction

Model output:

```python
start_logits
end_logits
```

Cần chọn answer tốt nhất từ tất cả features thuộc cùng một example.

Kỹ thuật:

- Với mỗi feature, lấy top `n_best` start positions và end positions.
- Bỏ span không hợp lệ:
  - không nằm trong context,
  - end < start,
  - quá dài.
- Score span = `start_logit + end_logit`.
- Chọn span score cao nhất.
- Convert token span về text bằng `offset_mapping` trên context gốc.

```python
n_best = 20
max_answer_length = 30
```

Metric:

```python
import evaluate

metric = evaluate.load("squad")
```

Kết quả gồm:

- `exact_match`
- `f1`

BERT fine-tuned đạt khoảng:

```text
exact_match: ~81.18
f1: ~88.67
```

## Training

```python
from transformers import AutoModelForQuestionAnswering

model = AutoModelForQuestionAnswering.from_pretrained("bert-base-cased")
```

Với QA, data đã padding max length nên có thể dùng default collator hoặc không cần custom collator.

TrainingArguments:

```python
from transformers import TrainingArguments

args = TrainingArguments(
    "bert-finetuned-squad",
    evaluation_strategy="no",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    fp16=True,
    push_to_hub=True,
)
```

`Trainer` không tiện compute metric thường xuyên vì QA post-processing cần thêm `features` và `examples`. Chương chỉ evaluate sau training, hoặc dùng custom loop với Accelerate để đánh giá mỗi epoch.

## Inference

```python
from transformers import pipeline

question_answerer = pipeline(
    "question-answering",
    model="huggingface-course/bert-finetuned-squad",
)

question_answerer(
    question="Which deep learning libraries back Transformers?",
    context=context,
)
```

---

# 7. Accelerate — các điểm kỹ thuật lặp lại

Nhiều phần có custom training loop với 🤗 Accelerate. Các pattern quan trọng:

## Prepare objects

```python
from accelerate import Accelerator

accelerator = Accelerator()

model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model,
    optimizer,
    train_dataloader,
    eval_dataloader,
)
```

Nếu dùng mixed precision:

```python
accelerator = Accelerator(fp16=True)
```

## Scheduler phải tạo sau `prepare()`

Vì `accelerator.prepare(train_dataloader)` có thể đổi length dataloader.

```python
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch
```

## Training loop cơ bản

```python
for epoch in range(num_train_epochs):
    model.train()

    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss

        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
```

## Gather trong distributed evaluation

Khi multi-GPU, cần:

```python
predictions = accelerator.gather(predictions)
labels = accelerator.gather(labels)
```

Nếu sequence length khác nhau giữa processes, cần pad trước:

```python
predictions = accelerator.pad_across_processes(
    predictions,
    dim=1,
    pad_index=-100,
)
```

## Save model đúng cách

```python
accelerator.wait_for_everyone()

unwrapped_model = accelerator.unwrap_model(model)

unwrapped_model.save_pretrained(
    output_dir,
    save_function=accelerator.save,
)

if accelerator.is_main_process:
    tokenizer.save_pretrained(output_dir)
```

Lý do:

- Model sau `prepare()` bị wrap.
- Cần unwrap để dùng `save_pretrained`.
- Chỉ main process save tokenizer/push hub.

---

# 8. Data collator cần nhớ theo task

| Task | Collator |
|---|---|
| Sequence classification | `DataCollatorWithPadding` |
| Token classification | `DataCollatorForTokenClassification` |
| MLM | `DataCollatorForLanguageModeling(mlm=True)` |
| Causal LM | `DataCollatorForLanguageModeling(mlm=False)` |
| Translation/Summarization | `DataCollatorForSeq2Seq` |
| QA đã padding max length | `default_data_collator` / `DefaultDataCollator` |

Điểm chung:

- Label padding thường là `-100`.
- `-100` được loss bỏ qua.
- Seq2Seq collator tạo `decoder_input_ids`.
- Causal LM labels là input copy, shift xảy ra trong model.

---

# 9. Metric cần nhớ theo task

| Task | Metric |
|---|---|
| Token classification | `seqeval`: precision, recall, F1, accuracy |
| MLM/CLM | Perplexity = `exp(loss)` |
| Translation | SacreBLEU |
| Summarization | ROUGE |
| Extractive QA | Exact Match, F1 |

Lưu ý:

- BLEU/ROUGE không đo hoàn hảo chất lượng ngôn ngữ.
- Translation/summarization có nhiều output đúng khác nhau.
- QA extractive cần post-processing phức tạp trước khi compute metric.

---

# 10. Khi nào fine-tune, domain-adapt, hoặc train từ đầu?

## Fine-tune task-specific

Dùng khi:

- Có pretrained model phù hợp.
- Có dataset cho task cụ thể.
- Muốn giải bài toán như NER, QA, translation, summarization.

## Domain adaptation

Dùng khi:

- Pretrained model chung không hiểu domain.
- Bạn có nhiều text trong domain nhưng ít label task-specific.
- Ví dụ legal, medical, scientific, movie reviews.

Pipeline:

```text
pretrained LM -> domain-adaptive LM fine-tuning -> task fine-tuning
```

## Train từ đầu

Chỉ nên khi:

- Không có pretrained model phù hợp với ngôn ngữ/domain.
- Dữ liệu rất khác text tự nhiên, ví dụ DNA, nhạc, code.
- Có đủ dữ liệu và compute.
- Cần kiểm soát bias/data source nghiêm ngặt.

---

# 11. Liên hệ với LLMs

Chương 7 là nền tảng kỹ thuật để hiểu LLMs:

- Tokenization.
- Pretraining objective: MLM, CLM.
- Fine-tuning.
- Domain adaptation.
- Seq2Seq generation.
- Evaluation.
- Distributed training.
- Inference bằng pipeline/generate.

LLMs hiện đại mở rộng các kỹ thuật này:

- Làm nhiều task không cần head riêng.
- Instruction following.
- Few-shot/zero-shot prompting.
- Chain-of-thought reasoning.
- Text generation đa mục đích.

Nhưng các khái niệm trong chương vẫn là lõi kỹ thuật để làm việc nghiêm túc với LLMs.