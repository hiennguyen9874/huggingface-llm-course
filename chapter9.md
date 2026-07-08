# 1. Gradio dùng để làm gì?

Gradio giúp bạn biến một hàm Python hoặc model ML thành một giao diện web tương tác chỉ với vài dòng code.

Ví dụ thay vì chỉ chạy model trong notebook:

```python
prediction = model(input)
```

Bạn có thể tạo UI để người khác nhập dữ liệu, bấm nút và xem kết quả ngay trên trình duyệt.

Gradio hữu ích cho:

- Demo model cho người không biết code.
- Chia sẻ model với team, khách hàng, researcher.
- Debug model bằng dữ liệu thực tế.
- Phát hiện lỗi, bias, edge case.
- Tạo demo trên web hoặc Hugging Face Spaces.

Gradio **không dùng để train model**, mà chủ yếu dùng cho **inference/demo** sau khi model đã được huấn luyện.

---

# 2. Cài đặt Gradio

```bash
pip install gradio
```

Gradio có thể chạy trong:

- Python script
- Jupyter Notebook
- Google Colab
- IDE thông thường
- Hugging Face Spaces

---

# 3. Demo đầu tiên với `gr.Interface`

Ví dụ đơn giản:

```python
import gradio as gr

def greet(name):
    return "Hello " + name

demo = gr.Interface(
    fn=greet,
    inputs="text",
    outputs="text"
)

demo.launch()
```

Ý nghĩa:

```python
gr.Interface(fn, inputs, outputs)
```

Trong đó:

| Tham số | Ý nghĩa |
|---|---|
| `fn` | Hàm Python được gọi khi người dùng submit |
| `inputs` | Loại input trên UI |
| `outputs` | Loại output trên UI |

Khi chạy:

```python
demo.launch()
```

Gradio sẽ mở web app tại:

```text
http://localhost:7860
```

Nếu chạy trong Notebook/Colab, UI có thể hiển thị trực tiếp trong notebook.

---

# 4. Custom component

Thay vì dùng shortcut `"text"`, có thể dùng class component để cấu hình rõ hơn.

```python
import gradio as gr

def greet(name):
    return "Hello " + name

textbox = gr.Textbox(
    label="Type your name here:",
    placeholder="John Doe",
    lines=2
)

gr.Interface(
    fn=greet,
    inputs=textbox,
    outputs="text"
).launch()
```

Điểm cần nhớ:

- `"text"` là shortcut.
- `gr.Textbox(...)` cho phép cấu hình label, placeholder, số dòng, v.v.

---

# 5. Kết hợp Gradio với Transformers model

Ví dụ demo text generation bằng GPT-2:

```python
from transformers import pipeline
import gradio as gr

model = pipeline("text-generation")

def predict(prompt):
    completion = model(prompt)[0]["generated_text"]
    return completion

gr.Interface(
    fn=predict,
    inputs="text",
    outputs="text"
).launch()
```

Luồng xử lý:

1. Người dùng nhập prompt.
2. Gradio gọi hàm `predict(prompt)`.
3. `predict` gọi model.
4. Kết quả trả về được render lên output textbox.

Gradio không phụ thuộc vào framework. Model có thể là:

- PyTorch
- TensorFlow
- scikit-learn
- Transformers
- Một API bên ngoài
- Một hàm Python bất kỳ

---

# 6. Hiểu kỹ `Interface`

Cấu trúc cơ bản:

```python
gr.Interface(
    fn=my_function,
    inputs=...,
    outputs=...
)
```

## Một input, một output

```python
def double(x):
    return x * 2

gr.Interface(
    fn=double,
    inputs="number",
    outputs="number"
).launch()
```

## Nhiều input

Khi hàm có nhiều tham số, truyền `inputs` dưới dạng list.

```python
import gradio as gr

def add(a, b):
    return a + b

gr.Interface(
    fn=add,
    inputs=[
        gr.Number(label="A"),
        gr.Number(label="B")
    ],
    outputs="number"
).launch()
```

Thứ tự rất quan trọng:

```python
inputs=[input_1, input_2, input_3]
```

sẽ map vào:

```python
fn(arg_1, arg_2, arg_3)
```

## Nhiều output

```python
def compute(x):
    return x * 2, x * 3

gr.Interface(
    fn=compute,
    inputs="number",
    outputs=[
        gr.Number(label="Double"),
        gr.Number(label="Triple")
    ]
).launch()
```

Thứ tự return cũng phải khớp với output components.

---

# 7. Làm việc với Audio

Ví dụ nhận audio từ microphone, đảo ngược âm thanh rồi phát lại:

```python
import numpy as np
import gradio as gr

def reverse_audio audio):
    sr, data = audio
    reversed_audio = (sr, np.flipud(data))
    return reversed_audio

mic = gr.Audio(
    source="microphone",
    type="numpy",
    label="Speak here..."
)

gr.Interface(
    fn=reverse_audio,
    inputs=mic,
    outputs="audio"
).launch()
```

Theo nội dung chương:

- `gr.Audio(type="numpy")` truyền vào hàm dạng tuple:

```python
(sample_rate, data)
```

- Output audio cũng có thể là tuple:

```python
(sample_rate, numpy_array)
```

Lưu ý: API Gradio mới có thể thay đổi tên tham số, nhưng concept vẫn là: audio có thể được truyền vào dưới dạng filepath hoặc numpy array.

---

# 8. Demo speech recognition

Ví dụ nhận audio rồi chuyển thành text:

```python
from transformers import pipeline
import gradio as gr

model = pipeline("automatic-speech-recognition")

def transcribe_audio(audio):
    transcription = model(audio)["text"]
    return transcription

gr.Interface(
    fn=transcribe_audio,
    inputs=gr.Audio(type="filepath"),
    outputs="text",
).launch()
```

Ở đây:

- Input audio được truyền vào hàm dưới dạng đường dẫn file.
- Pipeline ASR nhận file audio.
- Output là text transcription.

---

# 9. `launch()` và các tham số quan trọng

```python
demo.launch()
```

Một số tham số hay dùng:

| Tham số | Ý nghĩa |
|---|---|
| `inline=True/False` | Hiển thị inline trong notebook |
| `inbrowser=True` | Tự mở trình duyệt |
| `share=True` | Tạo public link tạm thời |
| `auth=...` | Thêm username/password authentication |

Ví dụ:

```python
demo.launch(
    share=True,
    inbrowser=True
)
```

---

# 10. Chia sẻ demo

Gradio hỗ trợ 2 cách chính.

## Cách 1: Temporary share link

```python
gr.Interface(
    fn=classify_image,
    inputs="image",
    outputs="label"
).launch(share=True)
```

Khi đó Gradio tạo public link dạng:

```text
xxxxx.gradio.app
```

Đặc điểm:

- Link public.
- Thường sống tối đa khoảng 72 giờ.
- Code/model vẫn chạy trên máy của bạn.
- Máy bạn phải bật.
- Không cần deploy phức tạp.

Rủi ro:

- Ai có link đều có thể dùng.
- Không nên expose function nguy hiểm.
- Không nên xử lý dữ liệu nhạy cảm nếu chưa kiểm soát bảo mật.

## Cách 2: Hugging Face Spaces

Hugging Face Spaces dùng để host demo lâu dài.

Thông thường repo Space có file:

```text
app.py
requirements.txt
```

Trong `app.py` bạn viết Gradio app:

```python
import gradio as gr

def greet(name):
    return "Hello " + name

gr.Interface(
    fn=greet,
    inputs="text",
    outputs="text"
).launch()
```

Spaces phù hợp khi muốn:

- Demo public lâu dài.
- Chia sẻ dễ.
- Không phụ thuộc máy cá nhân.
- Deploy cùng model, file config, dependency.

---

# 11. Làm đẹp demo bằng metadata

`Interface` hỗ trợ các tham số:

| Tham số | Ý nghĩa |
|---|---|
| `title` | Tiêu đề demo |
| `description` | Mô tả phía trên UI |
| `article` | Nội dung dài hơn phía dưới UI |
| `theme` | Theme giao diện |
| `examples` | Input mẫu |
| `live=True` | Tự chạy khi input thay đổi |

Ví dụ:

```python
title = "Ask Rick a Question"

description = """
The bot was trained to answer questions based on Rick and Morty dialogues.
Ask Rick anything!
"""

article = "Check out the original Rick and Morty Bot."

gr.Interface(
    fn=predict,
    inputs="textbox",
    outputs="text",
    title=title,
    description=description,
    article=article,
    examples=[
        ["What are you doing?"],
        ["Where should we time travel to?"]
    ],
).launch()
```

## `examples`

Nếu có một input:

```python
examples=[
    ["Hello"],
    ["Tell me a joke"]
]
```

Nếu có nhiều input:

```python
examples=[
    ["Hello", 0.7],
    ["Write a poem", 0.9]
]
```

Mỗi sample là một list, thứ tự tương ứng với input components.

## `live=True`

```python
gr.Interface(
    fn=predict,
    inputs="sketchpad",
    outputs="label",
    live=True
)
```

Khi `live=True`, hàm chạy lại mỗi khi input thay đổi, không cần bấm submit.

Phù hợp với:

- Sketch recognition
- Slider nhanh
- Preview realtime

Không phù hợp với:

- Model lớn, inference chậm
- API tốn tiền
- Tác vụ nặng

---

# 12. Ví dụ sketch recognition

Chương có ví dụ nhận hình vẽ và dự đoán top label.

Phần load label và model:

```python
from pathlib import Path
import torch
import gradio as gr
from torch import nn

LABELS = Path("class_names.txt").read_text().splitlines()

model = nn.Sequential(
    nn.Conv2d(1, 32, 3, padding="same"),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(32, 64, 3, padding="same"),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Conv2d(64, 128, 3, padding="same"),
    nn.ReLU(),
    nn.MaxPool2d(2),

    nn.Flatten(),
    nn.Linear(1152, 256),
    nn.ReLU(),
    nn.Linear(256, len(LABELS)),
)

state_dict = torch.load("pytorch_model.bin", map_location="cpu")
model.load_state_dict(state_dict, strict=False)
model.eval()
```

Hàm predict:

```python
def predict(im):
    x = torch.tensor(im, dtype=torch.float32)
    x = x.unsqueeze(0).unsqueeze(0) / 255.0

    with torch.no_grad():
        out = model(x)

    probabilities = torch.nn.functional.softmax(out[0], dim=0)
    values, indices = torch.topk(probabilities, 5)

    return {
        LABELS[i]: v.item()
        for i, v in zip(indices, values)
    }
```

Điểm kỹ thuật quan trọng:

```python
torch.tensor(im, dtype=torch.float32)
```

chuyển input image thành tensor.

```python
unsqueeze(0).unsqueeze(0)
```

thêm batch dimension và channel dimension.

Nếu ảnh grayscale shape ban đầu là:

```text
[H, W]
```

sau `unsqueeze` thành:

```text
[1, 1, H, W]
```

Tức là:

```text
[batch, channel, height, width]
```

Sau đó normalize:

```python
/ 255.0
```

đưa pixel từ `[0, 255]` về `[0, 1]`.

Dùng:

```python
model.eval()
```

để model ở inference mode.

Dùng:

```python
with torch.no_grad():
```

để không lưu gradient, tiết kiệm memory.

Output dạng dict:

```python
{
    "cat": 0.74,
    "dog": 0.15,
    "rabbit": 0.05
}
```

phù hợp với Gradio `"label"` output.

Interface:

```python
interface = gr.Interface(
    predict,
    inputs="sketchpad",
    outputs="label",
    theme="huggingface",
    title="Sketch Recognition",
    description="Draw a common object and the algorithm will guess in real time!",
    live=True,
)

interface.launch(share=True)
```

---

# 13. Tích hợp với Hugging Face Hub

Gradio có thể load trực tiếp model từ Hub thông qua:

```python
gr.Interface.load(...)
```

## Load model từ Hugging Face Hub

```python
import gradio as gr

gr.Interface.load(
    "huggingface/EleutherAI/gpt-j-6B",
    inputs=gr.Textbox(lines=5, label="Input Text"),
    title="GPT-J-6B"
).launch()
```

Có thể dùng prefix:

```python
"huggingface/user/model"
```

hoặc:

```python
"model/user/model"
```

Khi load kiểu này, Gradio dùng **Hugging Face Inference API**, không nhất thiết load model vào RAM local.

Điều này hữu ích với model lớn như GPT-J.

## Load Space từ Hugging Face Spaces

```python
gr.Interface.load(
    "spaces/abidlabs/remove-bg"
).launch()
```

Có thể override component hoặc metadata:

```python
gr.Interface.load(
    "spaces/abidlabs/remove-bg",
    inputs="webcam",
    title="Remove your webcam background!"
).launch()
```

Prefix hợp lệ:

```python
"spaces/user/space_name"
```

---

# 14. State trong Gradio

State dùng để lưu dữ liệu qua nhiều lần submit trong cùng một session.

Ví dụ thường gặp:

- Chatbot giữ lịch sử hội thoại.
- Form nhiều bước.
- Game state.
- Lưu context tạm thời.

State **không chia sẻ giữa nhiều user**.

## Ba bước thêm state

1. Hàm nhận thêm một tham số state.
2. Hàm trả thêm state mới.
3. Interface có thêm input `"state"` và output `"state"`.

Ví dụ chatbot:

```python
import random
import gradio as gr

def chat(message, history):
    history = history or []

    if message.startswith("How many"):
        response = random.randint(1, 10)
    elif message.startswith("How"):
        response = random.choice(["Great", "Good", "Okay", "Bad"])
    elif message.startswith("Where"):
        response = random.choice(["Here", "There", "Somewhere"])
    else:
        response = "I don't know"

    history.append((message, response))

    return history, history

iface = gr.Interface(
    fn=chat,
    inputs=["text", "state"],
    outputs=["chatbot", "state"],
    allow_screenshot=False,
    allow_flagging="never",
)

iface.launch()
```

Điểm quan trọng:

```python
return history, history
```

- Giá trị thứ nhất update chatbot UI.
- Giá trị thứ hai update state.

---

# 15. Interpretation: giải thích prediction

Gradio hỗ trợ interpretation để người dùng hiểu phần nào của input ảnh hưởng đến output.

Ví dụ image classification:

```python
import requests
import tensorflow as tf
import gradio as gr

inception_net = tf.keras.applications.MobileNetV2()

response = requests.get("https://git.io/JJkYN")
labels = response.text.split("\n")

def classify_image(inp):
    inp = inp.reshape((-1, 224, 224, 3))
    inp = tf.keras.applications.mobilenet_v2.preprocess_input(inp)
    prediction = inception_net.predict(inp).flatten()

    return {
        labels[i]: float(prediction[i])
        for i in range(1000)
    }

image = gr.Image(shape=(224, 224))
label = gr.Label(num_top_classes=3)

gr.Interface(
    fn=classify_image,
    inputs=image,
    outputs=label,
    interpretation="default",
    title="Image Classification + Interpretation"
).launch()
```

Các kiểu interpretation:

```python
interpretation="default"
```

hoặc:

```python
interpretation="shap"
```

Có thể chỉnh:

```python
num_shap=...
```

Hoặc truyền custom interpretation function.

---

# 16. `Interface` vs `Blocks`

Gradio có 2 API chính:

## `Interface`

High-level API.

Dùng khi:

- Một hàm chính.
- Input/output rõ ràng.
- Demo đơn giản.
- Muốn code nhanh.

Ví dụ:

```python
gr.Interface(
    fn=predict,
    inputs="text",
    outputs="text"
).launch()
```

## `Blocks`

Low-level API.

Dùng khi cần:

- Layout phức tạp.
- Nhiều tab.
- Nhiều button.
- Nhiều bước xử lý.
- Output của bước này làm input bước sau.
- Component thay đổi visibility, choices, properties.
- Kiểm soát event chi tiết.

---

# 17. Demo đầu tiên với `Blocks`

```python
import gradio as gr

def flip_text(x):
    return x[::-1]

demo = gr.Blocks()

with demo:
    gr.Markdown("""
    # Flip Text!
    Start typing below to see the output.
    """)

    input = gr.Textbox(placeholder="Flip this text")
    output = gr.Textbox()

    input.change(
        fn=flip_text,
        inputs=input,
        outputs=output
    )

demo.launch()
```

Ý tưởng chính:

- Tạo app bằng `gr.Blocks()`.
- Dùng `with demo:` để khai báo UI.
- Component được render theo thứ tự khai báo.
- Gắn event vào component.

Ở đây:

```python
input.change(...)
```

nghĩa là mỗi khi textbox thay đổi, gọi `flip_text`.

---

# 18. Event trong Blocks

Event có dạng:

```python
component.event(
    fn=...,
    inputs=...,
    outputs=...
)
```

Ví dụ:

```python
button.click(
    fn=predict,
    inputs=input_box,
    outputs=output_box
)
```

Một số event thường gặp:

| Event | Ý nghĩa |
|---|---|
| `.click()` | Khi button được click |
| `.change()` | Khi giá trị component thay đổi |
| `.submit()` | Khi submit textbox |
| `.play()` | Audio/video được play |
| `.pause()` | Audio/video pause |
| `.clear()` | Component bị clear |

Input/output có thể là:

```python
inputs=None
outputs=None
```

nếu function không nhận hoặc không trả dữ liệu.

---

# 19. Layout với Blocks

Mặc định các component xếp dọc.

## Row

```python
with gr.Row():
    input = gr.Textbox()
    output = gr.Textbox()
```

Các component trong `Row` xếp ngang.

## Column

```python
with gr.Column():
    input = gr.Textbox()
    output = gr.Textbox()
```

Các component trong `Column` xếp dọc.

## Tabs

```python
with gr.Tabs():
    with gr.TabItem("Tab 1"):
        ...

    with gr.TabItem("Tab 2"):
        ...
```

---

# 20. Ví dụ Blocks với nhiều tab

```python
import numpy as np
import gradio as gr

def flip_text(x):
    return x[::-1]

def flip_image(x):
    return np.fliplr(x)

with gr.Blocks() as demo:
    gr.Markdown("Flip text or image files using this demo.")

    with gr.Tabs():
        with gr.TabItem("Flip Text"):
            with gr.Row():
                text_input = gr.Textbox()
                text_output = gr.Textbox()
            text_button = gr.Button("Flip")

        with gr.TabItem("Flip Image"):
            with gr.Row():
                image_input = gr.Image()
                image_output = gr.Image()
            image_button = gr.Button("Flip")

    text_button.click(
        fn=flip_text,
        inputs=text_input,
        outputs=text_output
    )

    image_button.click(
        fn=flip_image,
        inputs=image_input,
        outputs=image_output
    )

demo.launch()
```

---

# 21. Input và output có thể là cùng một component

Ví dụ text completion: input textbox cũng là output textbox.

```python
import gradio as gr

api = gr.Interface.load("huggingface/EleutherAI/gpt-j-6B")

def complete_with_gpt(text):
    return text[:-50] + api(text[-50:])

with gr.Blocks() as demo:
    textbox = gr.Textbox(
        placeholder="Type here and press enter...",
        lines=4
    )
    btn = gr.Button("Generate")

    btn.click(
        fn=complete_with_gpt,
        inputs=textbox,
        outputs=textbox
    )

demo.launch()
```

Điểm quan trọng:

```python
outputs=textbox
```

nghĩa là kết quả ghi đè lại textbox ban đầu.

---

# 22. Multi-step demo với Blocks

Ví dụ:

1. Audio → speech-to-text.
2. Text → sentiment classification.

```python
from transformers import pipeline
import gradio as gr

asr = pipeline(
    "automatic-speech-recognition",
    "facebook/wav2vec2-base-960h"
)

classifier = pipeline("text-classification")

def speech_to_text(speech):
    text = asr(speech)["text"]
    return text

def text_to_sentiment(text):
    return classifier(text)[0]["label"]

with gr.Blocks() as demo:
    audio_file = gr.Audio(type="filepath")
    text = gr.Textbox()
    label = gr.Label()

    b1 = gr.Button("Recognize Speech")
    b2 = gr.Button("Classify Sentiment")

    b1.click(
        fn=speech_to_text,
        inputs=audio_file,
        outputs=text
    )

    b2.click(
        fn=text_to_sentiment,
        inputs=text,
        outputs=label
    )

demo.launch()
```

Điểm kỹ thuật quan trọng:

```python
text
```

vừa là output của bước 1, vừa là input của bước 2.

Đây là ưu điểm lớn của `Blocks`.

---

# 23. Update property của component

Không chỉ update value, Gradio còn có thể update property như:

- `visible`
- `lines`
- `choices`
- `interactive`
- `label`

Ví dụ thay đổi textbox theo radio:

```python
import gradio as gr

def change_textbox(choice):
    if choice == "short":
        return gr.Textbox.update(lines=2, visible=True)
    elif choice == "long":
        return gr.Textbox.update(lines=8, visible=True)
    else:
        return gr.Textbox.update(visible=False)

with gr.Blocks() as demo:
    radio = gr.Radio(
        ["short", "long", "none"],
        label="What kind of essay would you like to write?"
    )

    text = gr.Textbox(
        lines=2,
        interactive=True
    )

    radio.change(
        fn=change_textbox,
        inputs=radio,
        outputs=text
    )

demo.launch()
```

Ý tưởng:

- Nếu chọn `"short"`: textbox 2 dòng.
- Nếu chọn `"long"`: textbox 8 dòng.
- Nếu chọn `"none"`: ẩn textbox.

---

# 24. Các component quan trọng

Một số component được nhắc trong chương:

| Component | Công dụng |
|---|---|
| `gr.Textbox` | Nhập/xuất text |
| `gr.Number` | Nhập/xuất số |
| `gr.Slider` | Chọn số bằng thanh kéo |
| `gr.Dropdown` | Chọn từ danh sách |
| `gr.Radio` | Chọn một option |
| `gr.Image` | Upload hoặc hiển thị ảnh |
| `gr.Audio` | Upload/record/play audio |
| `gr.Label` | Hiển thị label + confidence |
| `gr.Chatbot` | Hiển thị hội thoại |
| `gr.Button` | Nút bấm |
| `gr.Markdown` | Nội dung markdown |
| `gr.Tabs`, `gr.TabItem` | Tabs |
| `gr.Row`, `gr.Column` | Layout |

---

# 25. Những điểm kỹ thuật cần nắm chắc

## 1. Function là trung tâm

Gradio bọc một hàm Python.

```python
def predict(input):
    return output
```

UI chỉ là cách đưa input vào hàm và hiển thị output.

---

## 2. Thứ tự input/output phải khớp

```python
def fn(a, b):
    return x, y
```

thì:

```python
inputs=[component_a, component_b]
outputs=[component_x, component_y]
```

---

## 3. Dữ liệu truyền vào phụ thuộc vào component

Ví dụ:

```python
gr.Audio(type="filepath")
```

truyền filepath.

```python
gr.Audio(type="numpy")
```

truyền tuple hoặc array dạng numpy.

```python
gr.Image()
```

thường truyền numpy array hoặc PIL image tùy cấu hình/API version.

---

## 4. Output dạng dict phù hợp với label

```python
return {
    "cat": 0.8,
    "dog": 0.2
}
```

dùng tốt với:

```python
outputs="label"
```

---

## 5. `Interface` nhanh, `Blocks` linh hoạt

Dùng `Interface` khi demo đơn giản.

Dùng `Blocks` khi có:

- Multi-step flow.
- Layout tùy chỉnh.
- Event phức tạp.
- Nhiều button.
- Nhiều tab.
- Shared intermediate component.

---

## 6. `share=True` là public

Không dùng cho dữ liệu nhạy cảm hoặc function nguy hiểm.

```python
demo.launch(share=True)
```

---

## 7. Hugging Face Spaces là cách deploy bền vững

Nếu muốn demo online lâu dài, dùng Spaces thay vì temporary link.

---

# 26. Checklist khi build một Gradio demo

1. Viết hàm inference:

```python
def predict(x):
    ...
    return y
```

2. Kiểm tra hàm chạy độc lập trước.

3. Chọn component input/output đúng kiểu dữ liệu.

4. Tạo `Interface` hoặc `Blocks`.

5. Thêm `title`, `description`, `examples`.

6. Test local:

```python
demo.launch()
```

7. Nếu cần chia sẻ nhanh:

```python
demo.launch(share=True)
```

8. Nếu cần host lâu dài: deploy lên Hugging Face Spaces.

---

# 27. Mẫu code tổng quát

## Với `Interface`

```python
import gradio as gr

def predict(x):
    result = x.upper()
    return result

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="Input"),
    outputs=gr.Textbox(label="Output"),
    title="Simple Demo",
    description="A minimal Gradio Interface example.",
    examples=[
        ["hello"],
        ["gradio"]
    ]
)

demo.launch()
```

## Với `Blocks`

```python
import gradio as gr

def predict(x):
    return x.upper()

with gr.Blocks() as demo:
    gr.Markdown("# Simple Blocks Demo")

    with gr.Row():
        inp = gr.Textbox(label="Input")
        out = gr.Textbox(label="Output")

    btn = gr.Button("Run")

    btn.click(
        fn=predict,
        inputs=inp,
        outputs=out
    )

demo.launch()
```

---

Tóm lại, chương 9 dạy cách dùng Gradio để biến model hoặc hàm Python thành demo web. Phần quan trọng nhất là hiểu `Interface`, `Blocks`, component input/output, event, state, cách chia sẻ bằng `share=True`, và cách deploy lâu dài bằng Hugging Face Spaces.