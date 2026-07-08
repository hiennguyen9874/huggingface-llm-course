# 1. Mục tiêu của Chapter 12

Chapter 12 giới thiệu cách dùng **Reinforcement Learning - RL** để huấn luyện LLM có khả năng **reasoning**, tức là biết suy luận nhiều bước trước khi trả lời.

Ví dụ thay vì model chỉ trả lời:

```text
5
```

Model reasoning có thể sinh:

```xml
<think>
I need to add 3 apples and 2 oranges.
</think>
5
```

Ý tưởng chính:

- LLM không chỉ học “dự đoán token tiếp theo”.
- LLM được thưởng/phạt dựa trên chất lượng câu trả lời.
- Với bài toán reasoning, reward có thể đến từ:
  - đáp án đúng/sai,
  - format đúng/sai,
  - độ dài phù hợp,
  - kiểm tra bằng rule,
  - đánh giá bởi model khác.

Chapter này xoay quanh **Open R1**, dự án cộng đồng của Hugging Face lấy cảm hứng từ DeepSeek R1, nhằm làm reasoning model dễ tiếp cận hơn.

---

# 2. Reinforcement Learning trong LLM

## 2.1. Các thành phần cơ bản của RL

Trong RL có 5 khái niệm chính:

| Thành phần | Trong RL nói chung | Trong LLM |
|---|---|---|
| Agent | Thực thể học hành động | Chính là language model |
| Environment | Môi trường phản hồi | Reward function, user, judge model, verifier |
| Action | Hành động agent chọn | Sinh token, sinh câu trả lời |
| Reward | Điểm thưởng/phạt | Điểm đánh giá response |
| Policy | Chiến lược chọn hành động | Phân phối xác suất sinh token của model |

Với LLM, **policy** chính là model hiện tại. Khi nhập prompt `q`, model sinh output `o` theo xác suất:

```text
πθ(o | q)
```

Trong đó:

- `πθ`: policy/model với tham số `θ`
- `q`: prompt
- `o`: completion/output

---

## 2.2. Quy trình RL

Một vòng RL cơ bản:

1. Model quan sát prompt.
2. Model sinh response.
3. Reward function chấm điểm response.
4. Model cập nhật trọng số để tăng xác suất sinh response tốt.
5. Lặp lại.

Với LLM, reward thường không đến trực tiếp từ “thế giới thật”, mà từ:

- human feedback,
- reward model,
- rule-based checker,
- unit test,
- exact match answer,
- format checker.

---

# 3. RLHF là gì?

**RLHF - Reinforcement Learning from Human Feedback** là kỹ thuật dùng phản hồi con người để align model.

Quy trình phổ biến:

## Bước 1: Thu thập preference

Cho cùng một prompt, model sinh nhiều câu trả lời. Human chọn câu nào tốt hơn.

Ví dụ:

```text
Prompt: Explain gravity to a child.

Answer A: ...
Answer B: ...

Human chọn B tốt hơn A.
```

## Bước 2: Train reward model

Dữ liệu preference được dùng để train một model riêng, gọi là **reward model**.

Reward model học cách chấm điểm response theo tiêu chí human thích.

## Bước 3: Fine-tune LLM bằng RL

LLM sinh response, reward model cho điểm, policy được cập nhật để sinh response có reward cao hơn.

---

# 4. PPO, DPO và GRPO khác nhau thế nào?

Chapter nhắc đến 3 hướng phổ biến:

## PPO - Proximal Policy Optimization

- Một thuật toán RLHF truyền thống.
- Dùng policy gradient.
- Thường cần reward model và value model/critic.
- Mạnh nhưng phức tạp, tốn tài nguyên.

## DPO - Direct Preference Optimization

- Đơn giản hơn PPO.
- Không cần reward model riêng.
- Dùng trực tiếp dữ liệu cặp `chosen/rejected`.
- Biến alignment thành bài toán phân loại preference.

## GRPO - Group Relative Policy Optimization

Đây là trọng tâm của chapter.

GRPO:

- Không bắt buộc cần reward model riêng.
- Sinh nhiều response cho cùng một prompt.
- So sánh các response trong cùng một group.
- Chuẩn hóa reward trong group.
- Cập nhật model dựa trên response nào tốt hơn trung bình group.

Điểm mạnh:

- Ổn định hơn do dùng so sánh theo nhóm.
- Phù hợp với bài toán có thể kiểm chứng như toán, code, logic.
- Có thể dùng rule-based reward.
- Không cần value model/critic như PPO, giảm chi phí tính toán.

---

# 5. DeepSeek R1 và các phát hiện chính

DeepSeek R1 là một bước tiến lớn trong reasoning model.

Chapter phân biệt hai model:

| Model | Cách train | Đặc điểm |
|---|---|---|
| DeepSeek-R1-Zero | Pure RL | Reasoning tự nổi lên, nhưng readability kém hơn |
| DeepSeek-R1 | SFT + RL nhiều pha | Reasoning tốt hơn, output dễ đọc hơn |

## 5.1. “Aha Moment”

Một phát hiện quan trọng của R1-Zero là model tự xuất hiện khả năng:

1. Thử lời giải ban đầu.
2. Nhận ra lỗi hoặc mâu thuẫn.
3. Tự sửa cách làm.
4. Giải thích vì sao cách mới tốt hơn.

Ví dụ tư duy:

```text
First attempt: This answer seems right.
Wait, I forgot to account for X.
Let me recompute.
Now the answer should be Y.
```

Điều quan trọng: khả năng này **không được lập trình trực tiếp**, mà xuất hiện từ quá trình RL.

---

# 6. Pipeline huấn luyện DeepSeek R1

DeepSeek R1 dùng 4 pha:

---

## 6.1. Cold Start Phase

Mục tiêu: tạo nền tảng output dễ đọc, có format tốt.

Cách làm:

- Lấy một tập nhỏ dữ liệu chất lượng cao từ R1-Zero.
- Dùng supervised fine-tuning trên DeepSeek-V3-Base.
- Giúp model ban đầu biết trình bày reasoning dễ đọc hơn.

Ý nghĩa kỹ thuật:

- Pure RL có thể tạo reasoning nhưng output có thể khó đọc.
- SFT ban đầu giúp ổn định format và readability.

---

## 6.2. Reasoning RL Phase

Mục tiêu: tăng năng lực reasoning cho toán, code, logic, science.

Đặc điểm:

- Dùng RL với reward dựa trên correctness.
- Tập trung vào task có thể kiểm chứng.
- Không cần reward model phức tạp.

Ví dụ:

- Toán: kiểm tra đáp án cuối.
- Code: chạy unit test.
- Logic: so khớp lời giải đúng.

Đây là pha quan trọng nhất để model học reasoning.

---

## 6.3. Rejection Sampling Phase

Mục tiêu: lọc output chất lượng cao.

Cách làm:

1. Model sinh nhiều sample.
2. DeepSeek-V3 hoặc judge khác đánh giá.
3. Giữ lại sample tốt.
4. Dùng dữ liệu đã lọc để SFT tiếp.

Ý nghĩa:

- Kết hợp khả năng sinh của model với quality control.
- Giảm output nhiễu hoặc sai format.

---

## 6.4. Diverse RL Phase

Mục tiêu: align model trên nhiều loại task hơn.

Reward kết hợp:

- Task deterministic: dùng rule-based reward.
- Task subjective: dùng LLM feedback.

Ví dụ:

| Task | Reward |
|---|---|
| Toán | Đúng đáp án |
| Code | Pass test |
| Chat/helpfulness | LLM judge |
| Format | Regex/rule |

---

# 7. GRPO hoạt động như thế nào?

GRPO gồm 3 phần chính:

1. Group sampling.
2. Advantage calculation.
3. Policy update.

---

## 7.1. Group Sampling

Với mỗi prompt `q`, model sinh `G` completions.

Ví dụ `G = 8`:

```text
Prompt: Calculate 2 + 2 * 6

o1 = 14
o2 = 16
o3 = 10
o4 = 14
o5 = 14
o6 = 12
o7 = 16
o8 = 14
```

Các output này được xem như một group.

Thông thường `G` là:

```text
4, 8 hoặc 16
```

Trade-off:

| Group size | Ưu điểm | Nhược điểm |
|---|---|---|
| Nhỏ 2-3 | Rẻ hơn | Ít diversity |
| 4-8 | Cân bằng tốt | Phù hợp đa số bài học |
| 8-16 | Tốt cho reasoning khó | Tốn compute hơn |
| Quá lớn | Có thể tốt hơn | Rất tốn VRAM/thời gian |

---

## 7.2. Reward Calculation

Mỗi output được chấm reward.

Ví dụ:

```text
Correct answer = 14

o1 = 14 -> reward = 1
o2 = 16 -> reward = 0
o3 = 10 -> reward = 0
o4 = 14 -> reward = 1
...
```

Reward có thể là:

```python
1.0 nếu đúng
0.0 nếu sai
```

Hoặc mềm hơn:

```python
2.0 nếu đúng hoàn toàn
0.5 nếu đúng format
0.5 nếu answer là integer
0.0 nếu sai
```

---

## 7.3. Advantage Calculation

GRPO không dùng reward tuyệt đối trực tiếp. Nó chuẩn hóa reward trong group.

Công thức:

```text
A_i = (r_i - mean(group_rewards)) / std(group_rewards)
```

Trong đó:

- `r_i`: reward của output thứ i.
- `mean`: reward trung bình trong group.
- `std`: độ lệch chuẩn reward trong group.
- `A_i`: advantage của output i.

Ý nghĩa:

- `A_i > 0`: output tốt hơn trung bình group.
- `A_i < 0`: output kém hơn trung bình group.
- `A_i = 0`: ngang trung bình.

Ví dụ:

```text
Rewards = [1, 0, 0, 1]
mean = 0.5
std ≈ 0.577

Correct output:
A = (1 - 0.5) / 0.577 ≈ 0.866

Wrong output:
A = (0 - 0.5) / 0.577 ≈ -0.866
```

Output đúng sẽ được tăng xác suất sinh, output sai bị giảm xác suất sinh.

---

# 8. Hàm mục tiêu của GRPO

GRPO cập nhật policy bằng objective kiểu PPO clipped objective, nhưng advantage được tính theo group.

Dạng đơn giản:

```text
maximize min(
    ratio * A,
    clip(ratio, 1 - ε, 1 + ε) * A
) - β * KL(policy || reference_policy)
```

Trong đó:

```text
ratio = πθ(o_i | q) / πold(o_i | q)
```

## 8.1. Probability Ratio

```text
ratio = new_probability / old_probability
```

Ý nghĩa:

- `ratio > 1`: model mới tăng xác suất output đó.
- `ratio < 1`: model mới giảm xác suất output đó.

Nếu output có advantage dương, ta muốn tăng xác suất nó.

Nếu output có advantage âm, ta muốn giảm xác suất nó.

---

## 8.2. Clipping

Clipping giới hạn update quá mạnh.

Ví dụ `ε = 0.2`:

```text
ratio chỉ được nằm trong [0.8, 1.2]
```

Nếu:

```text
old prob = 0.5
new prob = 0.9
ratio = 1.8
```

Sau clip:

```text
ratio = 1.2
```

Tác dụng:

- Tránh model thay đổi quá mạnh sau một step.
- Giữ training ổn định.
- Giảm nguy cơ reward hacking.

---

## 8.3. KL Penalty

GRPO dùng KL divergence để giữ policy mới không lệch quá xa khỏi reference model.

```text
β * D_KL(πθ || πref)
```

Ý nghĩa:

- `πθ`: model đang train.
- `πref`: reference/original model.
- `β`: hệ số kiểm soát mức phạt.

Nếu `β` cao:

- Model bị giữ gần model gốc.
- Training ổn định hơn.
- Nhưng học chậm hơn.

Nếu `β` thấp:

- Model học nhanh hơn.
- Nhưng dễ drift, sinh output kỳ lạ hoặc hack reward.

Trong DeepSeekMath, chapter nhắc `β = 0.04`.

---

# 9. Pseudocode GRPO

```text
Input:
- policy model
- reward function
- training prompts
- group size G

For each training step:
    For each prompt q:
        Generate G completions: o1, o2, ..., oG

        Compute rewards:
            r1, r2, ..., rG

        Compute group mean and std:
            mean_r = mean(rewards)
            std_r = std(rewards)

        Compute advantage:
            Ai = (ri - mean_r) / std_r

        Compute probability ratio:
            ratio = πθ(oi | q) / πold(oi | q)

        Optimize clipped objective:
            min(ratio * Ai, clip(ratio) * Ai)

        Add KL penalty against reference model

Output:
- optimized policy model
```

---

# 10. Ví dụ PyTorch tính advantage

```python
import torch

# 2 prompts, mỗi prompt có 4 generations
# group 1 rewards: [1, 0, 0, 1]
# group 2 rewards: [0, 0, 1, 1]
rewards = torch.tensor([1, 0, 0, 1, 0, 0, 1, 1], dtype=torch.float32)

num_generations = 4

# Shape: (B, G) = (2, 4)
rewards_grouped = rewards.view(-1, num_generations)

mean_grouped_rewards = rewards_grouped.mean(dim=1)
std_grouped_rewards = rewards_grouped.std(dim=1)

# Broadcast về shape (B * G,)
mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(num_generations)
std_grouped_rewards = std_grouped_rewards.repeat_interleave(num_generations)

advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-8)

print(advantages)
```

Output dạng:

```text
tensor([ 0.8660, -0.8660, -0.8660,  0.8660,
        -0.8660, -0.8660,  0.8660,  0.8660])
```

---

# 11. Implement GRPO bằng TRL

TRL cung cấp sẵn:

```python
from trl import GRPOTrainer, GRPOConfig
```

Các bước chính:

1. Load dataset prompt.
2. Định nghĩa reward function.
3. Tạo `GRPOConfig`.
4. Khởi tạo `GRPOTrainer`.
5. Train.

Ví dụ tối giản:

```python
from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset

dataset = load_dataset("your_dataset", split="train")

def reward_func(completions, **kwargs):
    return [float(len(completion)) for completion in completions]

training_args = GRPOConfig(
    output_dir="output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    logging_steps=10,
)

trainer = GRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    args=training_args,
    train_dataset=dataset,
    reward_funcs=reward_func,
)

trainer.train()
```

Lưu ý kỹ thuật: trong exercise thực tế dùng tham số `num_generations`.

```python
training_args = GRPOConfig(
    output_dir="output",
    num_generations=8,
    per_device_train_batch_size=4,
    learning_rate=1e-5,
)
```

---

# 12. Thiết kế reward function

Reward function là phần cực kỳ quan trọng. Model sẽ tối ưu đúng thứ bạn thưởng, kể cả khi đó không phải thứ bạn thật sự muốn.

## 12.1. Reward theo độ dài

```python
def reward_len(completions, **kwargs):
    ideal_length = 50
    return [-abs(ideal_length - len(completion)) for completion in completions]
```

Nếu completion dài đúng 50 token/ký tự hơn thì reward gần 0 hơn.

Ví dụ:

```text
len = 50 -> reward = 0
len = 40 -> reward = -10
len = 80 -> reward = -30
```

Dùng để ép model sinh output có độ dài mong muốn.

---

## 12.2. Reward theo format

Ví dụ muốn output có dạng:

```xml
<think>...</think>
<answer>...</answer>
```

Reward:

```python
import re

def format_reward(completions, **kwargs):
    pattern = r"<think>(.*?)</think>\s*<answer>(.*?)</answer>"

    rewards = []
    for completion in completions:
        match = re.search(pattern, completion, re.DOTALL)
        if match:
            think_content = match.group(1).strip()
            answer_content = match.group(2).strip()

            if len(think_content) > 20 and len(answer_content) > 0:
                rewards.append(1.0)
            else:
                rewards.append(0.5)
        else:
            rewards.append(0.0)

    return rewards
```

---

## 12.3. Reward cho bài toán có đáp án đúng

```python
def problem_reward(completions, answers, **kwargs):
    rewards = []

    for completion, correct_answer in zip(completions, answers):
        try:
            answer = extract_final_answer(completion)
            reward = 1.0 if answer == correct_answer else 0.0
        except Exception:
            reward = 0.0

        rewards.append(reward)

    return rewards
```

Phù hợp với:

- toán,
- code,
- câu hỏi có ground truth,
- bài benchmark có đáp án chuẩn.

---

# 13. Practical exercise: Fine-tune SmolLM bằng GRPO

Chapter có bài thực hành fine-tune model nhỏ với TRL.

## 13.1. Dependencies

```bash
pip install datasets==3.2.0 transformers==4.47.1 trl==0.14.0 peft==0.14.0 accelerate==1.2.1 bitsandbytes==0.45.2 wandb==0.19.7
pip install flash-attn --no-build-isolation
```

## 13.2. Model

Dùng model nhỏ:

```python
model_id = "HuggingFaceTB/SmolLM-135M-Instruct"
```

Load model:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2",
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
```

## 13.3. LoRA

Dùng LoRA để giảm số tham số cần train.

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

## 13.4. Reward length

```python
ideal_length = 50

def reward_len(completions, **kwargs):
    return [-abs(ideal_length - len(completion)) for completion in completions]
```

## 13.5. GRPOConfig

```python
from trl import GRPOConfig

training_args = GRPOConfig(
    output_dir="GRPO",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    max_prompt_length=512,
    max_completion_length=96,
    num_generations=8,
    optim="adamw_8bit",
    num_train_epochs=1,
    bf16=True,
    report_to=["wandb"],
    remove_unused_columns=False,
    logging_steps=1,
)
```

Các tham số quan trọng:

| Tham số | Ý nghĩa |
|---|---|
| `num_generations` | Số completion cho mỗi prompt |
| `max_prompt_length` | Độ dài prompt tối đa |
| `max_completion_length` | Độ dài completion tối đa |
| `per_device_train_batch_size` | Batch size trên mỗi GPU |
| `gradient_accumulation_steps` | Tích lũy gradient để giả lập batch lớn |
| `optim="adamw_8bit"` | Tiết kiệm memory |
| `bf16=True` | Dùng bfloat16 nếu GPU hỗ trợ |

## 13.6. Train

```python
from trl import GRPOTrainer

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[reward_len],
    args=training_args,
    train_dataset=dataset["train"],
)

trainer.train()
```

---

# 14. Cách đọc kết quả training GRPO

Chapter nhấn mạnh một điểm dễ gây nhầm:

## Reward tăng là dấu hiệu tốt

Với reward length:

```python
reward = -abs(ideal_length - len(completion))
```

Reward càng gần `0` càng tốt.

Ví dụ:

```text
length = 20, ideal = 50 -> reward = -30
length = 45, ideal = 50 -> reward = -5
length = 50, ideal = 50 -> reward = 0
```

## Loss có thể tăng, không nhất thiết là xấu

Trong supervised learning, loss tăng thường là xấu.

Nhưng trong GRPO, loss liên quan đến KL divergence và policy update. Khi model học tối ưu reward, nó có thể lệch khỏi initial policy, làm KL/loss tăng.

Vì vậy cần theo dõi đồng thời:

- reward,
- reward_std,
- kl,
- loss,
- output sample thực tế.

---

# 15. GRPO với Unsloth

Chapter cũng có bài thực hành dùng **Unsloth** để train nhanh hơn và ít tài nguyên hơn.

Dùng model:

```text
google/gemma-3-1b-it
```

## 15.1. Load model bằng Unsloth

```python
from unsloth import FastLanguageModel
import torch

max_seq_length = 1024
lora_rank = 32

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="google/gemma-3-1b-it",
    max_seq_length=max_seq_length,
    load_in_4bit=True,
    fast_inference=True,
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.6,
)
```

## 15.2. Apply LoRA

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=lora_rank,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)
```

Ý nghĩa:

- `load_in_4bit=True`: giảm VRAM.
- `fast_inference=True`: dùng vLLM để generate nhanh hơn.
- `use_gradient_checkpointing`: giảm memory khi train context dài.
- `target_modules`: các layer LoRA sẽ can thiệp.

---

# 16. Dataset GSM8K cho reasoning

Bài Unsloth dùng GSM8K, dataset toán tiểu học.

Prompt được ép theo format:

```python
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""
```

Format kỳ vọng:

```python
XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""
```

Chuẩn bị dataset:

```python
from datasets import load_dataset

def extract_hash_answer(text: str):
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def get_gsm8k_questions(split="train"):
    data = load_dataset("openai/gsm8k", "main")[split]
    data = data.map(
        lambda x: {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["question"]},
            ],
            "answer": extract_hash_answer(x["answer"]),
        }
    )
    return data

dataset = get_gsm8k_questions()
```

---

# 17. Multi-reward cho reasoning

Bài Unsloth dùng nhiều reward function cùng lúc.

## 17.1. Correctness reward

```python
def correctness_reward_func(prompts, completions, answer, **kwargs):
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]
```

Thưởng lớn nhất nếu đáp án đúng.

## 17.2. Integer reward

```python
def int_reward_func(completions, **kwargs):
    responses = [completion[0]["content"] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]
```

Thưởng nếu answer là số nguyên.

## 17.3. Strict format reward

```python
def strict_format_reward_func(completions, **kwargs):
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]
```

## 17.4. Soft format reward

```python
def soft_format_reward_func(completions, **kwargs):
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]
```

## 17.5. XML count reward

Thưởng từng phần nếu tag XML xuất hiện đúng một lần, đồng thời phạt text dư sau tag đóng.

```python
def count_xml(text) -> float:
    count = 0.0

    if text.count("<reasoning>\n") == 1:
        count += 0.125

    if text.count("\n</reasoning>\n") == 1:
        count += 0.125

    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001

    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001

    return count
```

Ý tưởng quan trọng: reward nên chia nhỏ để model học dần:

- đúng format,
- có answer,
- answer là số,
- answer đúng.

Nếu chỉ reward đúng/sai cuối cùng, tín hiệu học có thể quá sparse.

---

# 18. GRPOConfig với Unsloth

```python
from trl import GRPOConfig

training_args = GRPOConfig(
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    logging_steps=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_generations=6,
    max_prompt_length=256,
    max_completion_length=768,
    max_steps=250,
    save_steps=250,
    max_grad_norm=0.1,
    report_to="none",
    output_dir="outputs",
)
```

Khuyến nghị kỹ thuật:

- Nếu OOM, giảm `num_generations`.
- Nếu output reasoning dài, tăng `max_seq_length`.
- Nếu train không ổn định, giảm learning rate hoặc tăng gradient accumulation.
- Reward có thể chưa tăng trong 150-200 steps đầu.

---

# 19. Các giới hạn và rủi ro của GRPO

GRPO mạnh nhưng không miễn phí.

## 19.1. Chi phí generation cao

Với mỗi prompt, model phải sinh nhiều completion.

Nếu:

```text
batch_size = 8
num_generations = 8
```

Thực tế cần xử lý:

```text
64 completions mỗi step
```

Rất tốn VRAM và thời gian.

## 19.2. Reward design khó

Model sẽ tối ưu reward, không tối ưu “ý định thật” của bạn.

Ví dụ nếu reward là “càng dài càng tốt”, model có thể sinh text lan man.

Nếu reward chỉ check XML tag, model có thể học format đẹp nhưng answer sai.

## 19.3. Reward hacking

Nếu reward có lỗ hổng, model sẽ khai thác.

Ví dụ:

```python
reward = 1.0 if "42" in completion else 0.0
```

Model có thể luôn nhét `42` vào mọi câu.

## 19.4. KL tuning quan trọng

- KL quá mạnh: model không học được.
- KL quá yếu: model drift khỏi base model.

## 19.5. Group size trade-off

- Group nhỏ: reward normalization kém tin cậy.
- Group lớn: đắt compute.

---

# 20. Những điểm cần nắm thật chắc

Nếu chỉ nhớ các ý quan trọng nhất, nên nhớ:

1. **GRPO sinh nhiều response cho cùng một prompt**, rồi so sánh trong group.
2. **Advantage được chuẩn hóa trong group**:

   ```text
   A = (reward - group_mean) / group_std
   ```

3. **Không nhất thiết cần reward model**; reward có thể là rule, regex, unit test, exact match.
4. **Clipping** ngăn update quá lớn.
5. **KL penalty** giữ model không lệch quá xa model gốc.
6. **Reward function quyết định hành vi model học được**.
7. **GRPO phù hợp nhất với task kiểm chứng được**, như math/code/reasoning có đáp án rõ.
8. **Loss GRPO tăng không nhất thiết là xấu**; phải xem reward, KL và sample output.
9. **num_generations là tham số cốt lõi**, quyết định group size.
10. **Open R1/DeepSeek R1 cho thấy RL có thể tạo reasoning behavior**, kể cả “Aha Moment”.

---

# 21. Checklist khi tự train GRPO

Trước khi train:

```text
[ ] Dataset có cột prompt rõ ràng
[ ] Completion format mong muốn được định nghĩa
[ ] Reward function test được độc lập
[ ] Reward không quá dễ bị hack
[ ] num_generations phù hợp VRAM
[ ] max_prompt_length và max_completion_length đủ dài
[ ] Có log reward, reward_std, kl, loss
[ ] Có sample output định kỳ để kiểm tra thật
```

Với reasoning task:

```text
[ ] Có reward cho đúng đáp án
[ ] Có reward cho format
[ ] Có reward phụ để tránh sparse reward
[ ] Có parser robust để extract answer
[ ] Có phạt output dư/sai format nếu cần
```

---

# 22. Kết luận

Chapter 12 dạy rằng GRPO là một phương pháp thực dụng để huấn luyện LLM reasoning:

- đơn giản hơn PPO,
- linh hoạt hơn vì dùng được reward function bất kỳ,
- đặc biệt tốt cho bài toán có thể kiểm chứng,
- là kỹ thuật trung tâm đứng sau hướng Open R1/DeepSeek R1.

Nếu áp dụng thực tế, phần quan trọng nhất không phải chỉ là gọi `GRPOTrainer`, mà là **thiết kế reward function đúng**, chọn `num_generations` hợp lý, theo dõi KL/reward, và kiểm tra output thực tế thường xuyên.