# Chapter 11 — Supervised Fine-Tuning

Chương này nói về cách biến một language model đã pretrained thành một model hữu ích hơn cho hội thoại, làm theo hướng dẫn, sinh output theo format mong muốn, hoặc thích nghi với domain cụ thể.

Các phần chính:

1. Chat Templates  
2. Supervised Fine-Tuning, SFT  
3. LoRA, Low-Rank Adaptation  
4. Evaluation sau fine-tuning  

---

# 1. Chat Templates

## Chat template là gì?

Language model không “hiểu” hội thoại như con người. Về bản chất, model chỉ nhận một chuỗi token và dự đoán token tiếp theo.

Vì vậy, khi dùng model dạng chat/instruct, ta cần format đoạn hội thoại thành một chuỗi text theo đúng cấu trúc mà model đã được train. Cấu trúc đó gọi là **chat template**.

Ví dụ dữ liệu hội thoại thường được biểu diễn dạng:

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi! How can I help you today?"},
    {"role": "user", "content": "What's the weather?"},
]
```

Nhưng model không trực tiếp đọc list Python này. Tokenizer sẽ chuyển nó thành chuỗi text có special tokens phù hợp.

Ví dụ ChatML:

```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello!<|im_end|>
<|im_start|>assistant
Hi! How can I help you today?<|im_end|>
<|im_start|>user
What's the weather?<|im_start|>assistant
```

## Base model vs Instruct model

### Base model

Base model được train trên raw text để dự đoán token tiếp theo.

Ví dụ:

```text
HuggingFaceTB/SmolLM2-135M
```

Base model thường không tự nhiên biết trả lời kiểu chatbot. Nếu prompt không đúng format, nó có thể tiếp tục văn bản thay vì trả lời.

### Instruct model

Instruct model là base model đã được fine-tune để:

- làm theo instruction,
- trò chuyện theo role,
- trả lời hữu ích hơn,
- dùng tool/function calling,
- xử lý format hội thoại.

Ví dụ:

```text
HuggingFaceTB/SmolLM2-135M-Instruct
```

Điểm quan trọng: **instruct model thường phụ thuộc rất mạnh vào chat template đã dùng trong lúc training**.

Nếu dùng sai template, chất lượng output có thể giảm mạnh.

---

## Các format chat template phổ biến

Mỗi họ model có cách format khác nhau.

### ChatML, dùng bởi Qwen, SmolLM2

```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello!<|im_end|>
<|im_start|>assistant
Hi!<|im_end|>
```

### Mistral-style

```text
<s>[INST] You are a helpful assistant. [/INST]
Hi! How can I help you today?</s>
[INST] Hello! [/INST]
```

### Các khác biệt quan trọng

Cần chú ý:

1. **Role token**
   - system
   - user
   - assistant
   - tool

2. **Boundary token**
   - token bắt đầu message,
   - token kết thúc message,
   - token bắt đầu lượt assistant cần generate.

3. **System prompt handling**
   - Có model có role `system` riêng.
   - Có model nhét system prompt vào instruction đầu tiên.
   - Có model không hỗ trợ system message rõ ràng.

4. **Special tokens**
   - `<s>`
   - `</s>`
   - `[INST]`
   - `[/INST]`
   - `<|im_start|>`
   - `<|im_end|>`

---

## Dùng `apply_chat_template`

Transformers hỗ trợ tự động apply template theo tokenizer config của model.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    "HuggingFaceTB/SmolLM2-135M-Instruct"
)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

print(text)
```

Nếu muốn nhận token ids:

```python
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
)
```

`add_generation_prompt=True` thường thêm phần mở đầu cho assistant để model biết cần generate câu trả lời tiếp theo.

---

## Chat template cho tool use và multimodal

Chat template không chỉ dành cho text chat. Nó còn dùng cho:

- function calling,
- tool calling,
- image input,
- audio input,
- multi-turn conversation.

Ví dụ tool call:

```python
messages = [
    {
        "role": "system",
        "content": "You are an assistant that can use tools.",
    },
    {
        "role": "user",
        "content": "What is 123 * 456?",
    },
    {
        "role": "assistant",
        "content": "I will calculate it.",
        "tool_calls": [
            {
                "tool": "calculator",
                "parameters": {
                    "operation": "multiply",
                    "x": 123,
                    "y": 456,
                },
            }
        ],
    },
    {
        "role": "tool",
        "tool_name": "calculator",
        "content": "56088",
    },
]
```

Với multimodal:

```python
messages = [
    {
        "role": "system",
        "content": "You are a vision assistant.",
    },
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "What is in this image?"},
            {"type": "image", "image_url": "https://example.com/image.jpg"},
        ],
    },
]
```

---

## Best practices với chat template

Nên làm:

- Luôn dùng đúng template của model.
- Không trộn nhiều template trong cùng pipeline.
- Validate field `role`, `content`, `tool_calls`.
- Quản lý độ dài context, tránh vượt context window.
- Với tool/multimodal, test kỹ vì format khác nhau giữa các model.
- Kiểm tra `tokenizer_config.json` trên Hugging Face Hub để biết template chính xác.

Sai lầm phổ biến:

- Dùng template Mistral cho Qwen.
- Quên thêm generation prompt.
- Không escape special characters.
- Conversation history quá dài.
- Dataset training dùng một format, inference dùng format khác.

---

# 2. Supervised Fine-Tuning, SFT

## SFT là gì?

**Supervised Fine-Tuning** là quá trình fine-tune model pretrained bằng các cặp input-output có nhãn.

Mục tiêu là dạy model:

- làm theo instruction,
- trả lời như assistant,
- dùng đúng tone/style,
- sinh output theo schema cố định,
- hiểu domain cụ thể,
- tuân thủ chat template.

Ví dụ dữ liệu SFT:

```python
{
    "messages": [
        {"role": "user", "content": "Explain LoRA simply."},
        {"role": "assistant", "content": "LoRA is a parameter-efficient..."}
    ]
}
```

Hoặc dạng input-output:

```python
{
    "input": "Summarize this article...",
    "output": "The article says..."
}
```

---

## Khi nào nên dùng SFT?

Không phải lúc nào cũng cần SFT.

Trước tiên nên thử:

- prompt engineering,
- few-shot prompting,
- dùng model instruct tốt hơn,
- retrieval-augmented generation, RAG.

Chỉ nên dùng SFT nếu:

1. Prompting không đủ tốt.
2. Cần output format cực kỳ ổn định.
3. Model phải theo style/domain riêng.
4. Muốn dùng model nhỏ hơn để giảm chi phí inference.
5. Có dữ liệu chất lượng cao.

Ví dụ nên SFT:

- model sinh JSON theo schema nội bộ,
- chatbot ngành y/law/finance cần terminology chuẩn,
- assistant hỗ trợ khách hàng theo policy công ty,
- model code theo convention riêng.

Ví dụ không nên SFT ngay:

- chỉ cần model trả lời tốt hơn một chút,
- chưa có dataset chất lượng,
- task thay đổi liên tục,
- lỗi thực ra đến từ thiếu context, nên dùng RAG.

---

## Chuẩn bị dataset cho SFT

Dataset cần có:

1. Prompt hoặc user message.
2. Expected response.
3. Optional context/metadata.
4. Format nhất quán với chat template.

Ví dụ convert dataset input-output sang messages:

```python
def convert_to_chatml(example):
    return {
        "messages": [
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["output"]},
        ]
    }
```

Với Hugging Face Datasets:

```python
from datasets import load_dataset

dataset = load_dataset("HuggingFaceTB/smoltalk", "all")

dataset = dataset.map(convert_to_chatml)
```

Nếu dataset đã có field `messages`, `SFTTrainer` có thể tự apply chat template bằng tokenizer.

---

## Các tham số training quan trọng

### 1. Thời lượng training

```python
num_train_epochs=3
```

hoặc:

```python
max_steps=1000
```

- Train quá ít: underfit.
- Train quá nhiều: overfit hoặc memorization.

### 2. Batch size

```python
per_device_train_batch_size=4
```

Batch lớn hơn giúp gradient ổn định hơn, nhưng tốn VRAM hơn.

### 3. Gradient accumulation

```python
gradient_accumulation_steps=8
```

Effective batch size:

```text
effective_batch_size =
    per_device_train_batch_size
    * gradient_accumulation_steps
    * number_of_devices
```

Ví dụ:

```text
4 * 8 * 1 = 32
```

Nó giúp giả lập batch lớn mà không cần nhiều VRAM.

### 4. Learning rate

```python
learning_rate=5e-5
```

- Quá cao: loss dao động, training unstable.
- Quá thấp: học chậm hoặc không học.

### 5. Warmup

```python
warmup_ratio=0.03
```

Warmup giúp tăng learning rate từ từ ở đầu training, tránh update quá mạnh khi model chưa ổn định.

### 6. Logging, eval, checkpoint

```python
logging_steps=10
eval_steps=50
save_steps=100
```

Dùng để theo dõi loss, eval định kỳ, lưu checkpoint.

---

## Ví dụ SFT với TRL

Một pipeline SFT cơ bản:

```python
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from trl.models.utils import setup_chat_format

model_name = "HuggingFaceTB/SmolLM2-135M"

dataset = load_dataset("HuggingFaceTB/smoltalk", "all")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

model, tokenizer = setup_chat_format(
    model=model,
    tokenizer=tokenizer,
)

training_args = SFTConfig(
    output_dir="./sft_output",
    max_steps=1000,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    logging_steps=10,
    save_steps=100,
    eval_strategy="steps",
    eval_steps=50,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
)

trainer.train()
```

Lưu ý kỹ thuật:

- Nếu dataset có `messages`, trainer có thể tự apply chat template.
- Nếu dataset là `question/answer`, cần `formatting_func`.
- Nếu tokenizer thiếu `pad_token`, thường cần set:

```python
tokenizer.pad_token = tokenizer.eos_token
```

---

## Dataset packing

Nhiều example SFT rất ngắn. Nếu mỗi example chiếm một sequence riêng, GPU bị lãng phí.

**Packing** cho phép ghép nhiều sample ngắn vào cùng một sequence dài hơn.

```python
training_args = SFTConfig(
    output_dir="./sft_output",
    packing=True,
)
```

Ưu điểm:

- tận dụng context length tốt hơn,
- tăng throughput,
- giảm padding.

Rủi ro:

- khi dùng `max_steps`, số epoch thực tế có thể khác dự kiến,
- cần cẩn thận với boundary giữa các example,
- không phải mọi eval đều nên packing.

Có thể tắt packing cho eval:

```python
SFTConfig(
    packing=True,
    eval_packing=False,
)
```

Với dataset custom:

```python
def formatting_func(example):
    return f"### Question: {example['question']}\n### Answer: {example['answer']}"
```

---

## Theo dõi training

Các metric nên xem:

- training loss,
- validation loss,
- learning rate,
- gradient norm,
- output thực tế của model.

### Loss pattern tốt

Thông thường:

1. Loss giảm mạnh lúc đầu.
2. Sau đó giảm chậm.
3. Cuối cùng ổn định.

Dấu hiệu tốt:

```text
training loss giảm
validation loss giảm
khoảng cách giữa hai loss nhỏ
```

### Overfitting

Dấu hiệu:

```text
training loss giảm
validation loss tăng
```

Cách xử lý:

- giảm số step/epoch,
- tăng dataset,
- tăng data diversity,
- dùng LoRA dropout,
- regularization,
- kiểm tra leakage giữa train/eval.

### Underfitting

Dấu hiệu:

```text
loss gần như không giảm
output vẫn kém
```

Có thể thử:

- tăng learning rate,
- train lâu hơn,
- kiểm tra dataset format,
- kiểm tra chat template,
- dùng model lớn hơn.

### Memorization

Dấu hiệu:

```text
loss cực thấp
model lặp lại training examples
output thiếu đa dạng
perform kém trên input mới
```

Cần chú ý: loss đẹp không đảm bảo model tốt. Phải kiểm tra qualitative output.

---

# 3. LoRA, Low-Rank Adaptation

## Vấn đề của full fine-tuning

Full fine-tuning cập nhật toàn bộ trọng số model.

Với LLM lớn, điều này tốn:

- VRAM,
- thời gian,
- checkpoint storage,
- compute.

Ví dụ model vài tỷ parameter có thể cần nhiều GPU nếu full fine-tune.

---

## LoRA là gì?

**LoRA** là kỹ thuật parameter-efficient fine-tuning.

Ý tưởng chính:

- Freeze trọng số gốc của model.
- Không update matrix trọng số lớn `W`.
- Thêm hai matrix nhỏ, trainable, vào một số layer.
- Chỉ train các adapter nhỏ đó.

Nếu layer gốc có weight:

```text
W
```

LoRA học phần update:

```text
ΔW = B @ A
```

Trong đó:

```text
A: low-rank matrix
B: low-rank matrix
rank r nhỏ hơn nhiều so với dimension gốc
```

Forward logic về mặt ý tưởng:

```text
output = W x + scale * B A x
```

Thay vì train toàn bộ `W`, ta chỉ train `A` và `B`.

---

## Vì sao LoRA tiết kiệm?

Nếu weight gốc có kích thước:

```text
d x k
```

Full fine-tuning cần train:

```text
d * k parameters
```

LoRA với rank `r` chỉ train:

```text
r * k + d * r
```

Nếu `r` nhỏ, số parameter giảm rất mạnh.

Ví dụ:

```text
d = 4096
k = 4096
r = 8
```

Full:

```text
4096 * 4096 = 16,777,216 parameters
```

LoRA:

```text
8 * 4096 + 4096 * 8 = 65,536 parameters
```

Giảm khoảng 256 lần cho layer đó.

---

## Ưu điểm LoRA

1. Tiết kiệm VRAM.
2. Train nhanh hơn full fine-tuning.
3. Checkpoint nhỏ vì chỉ lưu adapter.
4. Có thể switch nhiều adapter cho nhiều task.
5. Có thể merge adapter vào base model khi deploy.
6. Kết hợp được với quantization thành QLoRA.

---

## Các tham số LoRA quan trọng

```python
from peft import LoraConfig

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)
```

### `r`

Rank của low-rank matrices.

Thường dùng:

```text
4, 8, 16, 32
```

- `r` nhỏ: tiết kiệm hơn, ít expressive hơn.
- `r` lớn: học tốt hơn, tốn VRAM hơn.

### `lora_alpha`

Scaling factor.

Thường đặt khoảng:

```text
2 * r
```

Scale thường là:

```text
lora_alpha / r
```

Nó kiểm soát mức ảnh hưởng của LoRA adapter.

### `lora_dropout`

Dropout cho LoRA path.

Thường:

```text
0.05 - 0.1
```

Giúp giảm overfitting.

### `bias`

```python
bias="none"
```

Phổ biến nhất vì tiết kiệm.

Các option:

- `"none"`
- `"all"`
- `"lora_only"`

### `target_modules`

Chọn module nào được gắn LoRA.

Ví dụ:

```python
target_modules="all-linear"
```

hoặc cụ thể:

```python
target_modules=["q_proj", "v_proj"]
```

Nếu áp dụng nhiều module hơn:

- model thích nghi tốt hơn,
- nhưng tốn bộ nhớ hơn.

---

## Fine-tune SFT với LoRA

Ví dụ tích hợp `SFTTrainer` và PEFT:

```python
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

dataset = load_dataset("HuggingFaceTB/smoltalk", "all")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

args = SFTConfig(
    output_dir="./lora_sft_output",
    max_steps=1000,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    logging_steps=10,
    save_steps=100,
    eval_strategy="steps",
    eval_steps=100,
    packing=True,
)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
    peft_config=peft_config,
)

trainer.train()

trainer.save_model("./lora_adapter")
```

Điểm đáng chú ý:

- Learning rate LoRA thường có thể cao hơn full fine-tuning.
- Output lưu ra thường là adapter, không phải full model.
- Base model vẫn cần có khi inference nếu chưa merge.

---

## Load LoRA adapter

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

adapter_path = "./lora_adapter"

config = PeftConfig.from_pretrained(adapter_path)

base_model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    device_map="auto",
)

model = PeftModel.from_pretrained(
    base_model,
    adapter_path,
)

tokenizer = AutoTokenizer.from_pretrained(
    config.base_model_name_or_path
)
```

---

## Merge LoRA adapter vào base model

Khi deploy, có thể merge adapter vào model gốc để không cần load adapter riêng.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
adapter_path = "./lora_adapter"
merged_path = "./merged_model"

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="auto",
)

peft_model = PeftModel.from_pretrained(
    base_model,
    adapter_path,
    torch_dtype=torch.float16,
)

merged_model = peft_model.merge_and_unload()

tokenizer = AutoTokenizer.from_pretrained(base_model_name)

merged_model.save_pretrained(merged_path)
tokenizer.save_pretrained(merged_path)
```

Sau merge:

- model trở thành một model bình thường,
- inference đơn giản hơn,
- không cần adapter riêng,
- nhưng mất tính linh hoạt switch adapter.

---

# 4. Evaluation

Fine-tune xong không có nghĩa là model tốt. Phải evaluate.

Chương này nhấn mạnh: **automatic benchmark chỉ là một phần của evaluation strategy**.

---

## Automatic benchmarks

Benchmark tự động là các bộ test chuẩn với metric rõ ràng.

Ưu điểm:

- kết quả reproducible,
- dễ so sánh giữa model,
- có sẵn nhiều task,
- chạy tự động.

Nhược điểm:

- không luôn phản ánh performance thực tế,
- có thể không khớp domain riêng,
- có thể bị model “học thuộc” benchmark,
- không đánh giá đầy đủ safety, style, UX.

---

## Một số benchmark phổ biến

### MMLU

Massive Multitask Language Understanding.

- 57 subjects.
- Bao gồm science, humanities, medicine, law...
- Đánh giá kiến thức rộng.

### TruthfulQA

Đánh giá model có hay lặp lại misconception hoặc thông tin sai phổ biến không.

### BBH

Big Bench Hard.

- Tập trung vào reasoning khó.
- Logic, planning, multi-step reasoning.

### GSM8K

- Toán lời văn cấp phổ thông.
- Đánh giá arithmetic reasoning.

### MATH

- Bài toán thi cạnh tranh.
- Cần multi-step reasoning.
- Gồm algebra, geometry, number theory, probability...

### HumanEval

- Benchmark code generation.
- Có test cases chạy thật.
- Đánh giá functional correctness, không chỉ text similarity.

### AlpacaEval

- Đánh giá instruction-following.
- Thường dùng LLM mạnh như GPT-4 làm judge.
- Tập trung helpfulness, honesty, harmlessness.

---

## LLM-as-Judge

Dùng một LLM để chấm output của LLM khác.

Ưu điểm:

- scalable,
- đánh giá được tiêu chí mềm như helpfulness, clarity, completeness,
- nhanh hơn human review.

Rủi ro:

- judge model có bias,
- có thể ưu tiên văn phong hơn độ đúng,
- prompt judge ảnh hưởng kết quả,
- cần kiểm tra agreement với human/domain expert.

---

## Evaluation arenas

Ví dụ Chatbot Arena.

Cách hoạt động:

- người dùng thấy hai câu trả lời ẩn danh,
- vote câu nào tốt hơn,
- hệ thống tính ranking.

Ưu điểm:

- gần với use case thực tế,
- đa dạng prompt,
- phản ánh preference người dùng.

Nhược điểm:

- user distribution bias,
- prompt không đại diện cho domain cụ thể,
- thường tập trung helpfulness hơn safety/compliance.

---

## Custom evaluation

Với sản phẩm thực tế, nên tạo benchmark riêng.

Nên bao gồm:

1. Real user queries.
2. Edge cases.
3. Các lỗi nghiêm trọng cần tránh.
4. Domain-specific cases.
5. Format/schema tests.
6. Safety/compliance tests.
7. Regression set cho lỗi từng gặp.

Ví dụ nếu fine-tune model tạo JSON:

```python
import json

def is_valid_json_output(text):
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return False

    required_keys = {"title", "price", "description"}
    return required_keys.issubset(obj.keys())
```

Nếu model dùng cho y tế:

- kiểm tra kiến thức medical,
- kiểm tra hallucination,
- kiểm tra disclaimer,
- kiểm tra referral khi nguy hiểm,
- domain expert review.

---

## Multi-layer evaluation strategy

Một evaluation tốt nên có nhiều lớp:

1. **Standard benchmarks**
   - MMLU, GSM8K, HumanEval...

2. **Custom benchmark**
   - dữ liệu gần use case thật.

3. **Qualitative review**
   - đọc output thực tế.

4. **Human/domain expert review**
   - đặc biệt với domain rủi ro cao.

5. **A/B testing**
   - thử trong môi trường kiểm soát.

6. **Production monitoring**
   - track complaint, fallback, latency, cost, hallucination reports.

---

## Dùng LightEval

`lighteval` là tool để chạy benchmark.

Format task:

```text
{suite}|{task}|{num_few_shot}|{auto_reduce}
```

Ví dụ:

```text
mmlu|abstract_algebra|0|0
```

Ý nghĩa:

- `suite`: benchmark suite, ví dụ `mmlu`.
- `task`: task cụ thể.
- `num_few_shot`: số example few-shot trong prompt.
- `auto_reduce`: có tự giảm few-shot nếu prompt quá dài không.

Ví dụ evaluate model trên các task liên quan medicine:

```bash
lighteval accelerate \
    "pretrained=your-model-name" \
    "mmlu|anatomy|0|0" \
    "mmlu|high_school_biology|0|0" \
    "mmlu|high_school_chemistry|0|0" \
    "mmlu|professional_medicine|0|0" \
    --max_samples 40 \
    --batch_size 1 \
    --output_path "./results" \
    --save_generations true
```

Kết quả thường có dạng bảng:

```text
| Task                             | Metric | Value |
|----------------------------------|--------|-------|
| mmlu:anatomy                     | acc    | 0.45  |
| mmlu:high_school_biology         | acc    | 0.15  |
| mmlu:professional_medicine       | acc    | 0.33  |
```

---

# Quy trình thực tế nên dùng

Một pipeline fine-tuning ổn thường như sau:

## Bước 1: Chọn model nền

Ví dụ:

```text
deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
HuggingFaceTB/SmolLM2-135M
Qwen/Qwen2.5-1.5B-Instruct
```

Cần kiểm tra:

- license,
- context length,
- tokenizer,
- chat template,
- VRAM requirement,
- model đã instruct chưa.

---

## Bước 2: Chuẩn hóa dataset

Nên đưa về dạng:

```python
{
    "messages": [
        {"role": "system", "content": "... optional ..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]
}
```

Kiểm tra:

- không có response rỗng,
- role hợp lệ,
- không bị duplicate quá nhiều,
- không lẫn eval vào train,
- output đúng format mong muốn.

---

## Bước 3: Chọn phương pháp fine-tune

Nếu model nhỏ và có nhiều compute:

```text
full SFT
```

Nếu model lớn hoặc GPU hạn chế:

```text
LoRA SFT
```

Nếu cần cực kỳ tiết kiệm VRAM:

```text
QLoRA
```

---

## Bước 4: Train thử nhỏ

Không nên train full ngay.

Nên chạy:

```text
100 - 500 steps
```

Để kiểm tra:

- dataset format đúng không,
- loss có giảm không,
- model có generate đúng template không,
- không bị lỗi tokenizer/padding.

---

## Bước 5: Train chính thức

Theo dõi:

- train loss,
- eval loss,
- samples output,
- checkpoint.

---

## Bước 6: Evaluate

Chạy:

- benchmark chuẩn,
- custom eval,
- qualitative review.

---

## Bước 7: Deploy hoặc merge adapter

Nếu dùng LoRA:

- giữ adapter riêng nếu muốn switch task,
- merge nếu muốn deploy đơn giản.

---

# Những điểm kỹ thuật cần nhớ nhất

1. **Chat template phải khớp giữa training và inference.**
2. **SFT không thay thế dữ liệu chất lượng. Dataset kém thì model kém.**
3. **Không nên SFT nếu prompting hoặc RAG đã đủ.**
4. **Validation loss tăng trong khi train loss giảm là dấu hiệu overfitting.**
5. **Gradient accumulation giúp tăng effective batch size mà không tăng VRAM nhiều.**
6. **Packing giúp tận dụng GPU tốt hơn nhưng cần hiểu tác động lên số step/epoch.**
7. **LoRA freeze base model và chỉ train adapter nhỏ.**
8. **LoRA rank `r` càng lớn thì càng expressive nhưng tốn hơn.**
9. **Adapter LoRA có thể load riêng hoặc merge vào base model.**
10. **Benchmark chuẩn không đủ; cần custom evaluation theo use case thật.**
11. **Loss đẹp không đảm bảo output tốt; luôn kiểm tra qualitative output.**
12. **Với domain rủi ro cao, cần human/domain expert review.**

---

Tóm lại, Chapter 11 dạy quy trình biến model pretrained thành assistant/task-specific model bằng SFT, tối ưu tài nguyên bằng LoRA, và đánh giá model bằng cả benchmark chuẩn lẫn custom evaluation.