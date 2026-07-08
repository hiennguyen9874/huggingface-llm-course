# Chapter 10 — Argilla: Annotate và curate dataset

## 1. Argilla dùng để làm gì?

Argilla là một nền tảng giúp bạn quản lý quy trình **data annotation** và **dataset curation**.

Nói ngắn gọn: nếu mô hình của bạn cần dữ liệu chất lượng cao, Argilla giúp bạn biến dữ liệu thô hoặc dữ liệu gán nhãn chưa tốt thành dataset sạch hơn, nhất quán hơn và dùng được cho training/evaluation.

Argilla có thể dùng để:

- Biến dữ liệu chưa có cấu trúc thành dữ liệu có cấu trúc cho NLP.
- Gán nhãn dữ liệu cho các task như:
  - Text classification
  - Token classification / Named Entity Recognition
  - Human feedback cho LLM
  - Multimodal annotation
- Rà soát và chỉnh sửa nhãn có sẵn.
- Làm việc nhóm với nhiều annotator.
- Xuất dataset đã annotate lên Hugging Face Hub.

Điểm quan trọng: **Argilla không trực tiếp train model**. Nó giúp bạn tạo hoặc cải thiện dataset, sau đó dataset đó mới được dùng để train model bằng thư viện khác như `transformers`, `datasets`, `setfit`, v.v.

---

# 2. Thiết lập Argilla instance

Muốn dùng Argilla, bạn cần có một Argilla server/instance đang chạy, rồi kết nối tới nó bằng Python SDK.

## 2.1. Deploy Argilla UI

Cách đơn giản nhất trong khóa học là deploy Argilla bằng **Hugging Face Spaces**.

Bạn tạo Space từ template Argilla:

```text
https://huggingface.co/new-space?template=argilla%2Fargilla-template-space
```

Sau khi Space chạy, bạn sẽ có một giao diện web để đăng nhập và quản lý dataset.

### Lưu ý quan trọng

Nếu dùng Hugging Face Space, nên bật:

```text
Persistent storage
```

Nếu không bật, dữ liệu có thể bị mất khi Space bị pause hoặc restart.

---

## 2.2. Cài Python SDK

Trong notebook hoặc môi trường Python:

```python
!pip install argilla
```

Import thư viện:

```python
import argilla as rg
```

---

## 2.3. Kết nối tới Argilla server

Bạn cần các thông tin sau:

### API URL

Nếu dùng Hugging Face Space, API URL thường có dạng:

```text
https://<your-username>.<space-name>.hf.space
```

Bạn có thể lấy URL này bằng cách mở Space, chọn dấu ba chấm, rồi chọn **Embed this Space**, sau đó copy **Direct URL**.

### API key

Vào Argilla UI:

```text
My Settings -> API key
```

### Hugging Face token

Chỉ cần nếu Space là private.

Ví dụ kết nối:

```python
import argilla as rg

HF_TOKEN = "..."  # chỉ cần nếu Space private

client = rg.Argilla(
    api_url="https://<your-username>.<space-name>.hf.space",
    api_key="...",
    headers={"Authorization": f"Bearer {HF_TOKEN}"},  # chỉ cần nếu private
)
```

Kiểm tra kết nối:

```python
client.me
```

Nếu trả về thông tin user, nghĩa là SDK đã kết nối thành công.

---

# 3. Load dataset vào Argilla

Trong chương này, ví dụ dùng dataset:

```text
SetFit/ag_news
```

Đây là dataset tin tức, dùng cho hai task:

1. **Text classification**: phân loại chủ đề của bài viết.
2. **Token classification / NER**: đánh dấu named entities như PERSON, ORG, LOC, EVENT.

---

## 3.1. Load dataset từ Hugging Face Hub

```python
from datasets import load_dataset

data = load_dataset("SetFit/ag_news", split="train")
data.features
```

Output có dạng:

```python
{
    "text": Value(dtype="string"),
    "label": Value(dtype="int64"),
    "label_text": Value(dtype="string")
}
```

Dataset có:

- `text`: nội dung tin tức.
- `label`: nhãn dạng số.
- `label_text`: nhãn dạng chữ, ví dụ chủ đề của tin tức.

---

# 4. Cấu hình dataset trong Argilla

Trong Argilla, trước khi upload records, bạn cần định nghĩa **settings** cho dataset.

Settings gồm hai phần chính:

## 4.1. Fields

`fields` là dữ liệu được hiển thị cho annotator xem và gán nhãn.

Ví dụ:

```python
fields=[rg.TextField(name="text")]
```

Ở đây dataset có một trường text, nên dùng `TextField`.

Argilla hỗ trợ nhiều loại field khác nhau như text, chat, image, v.v. tùy bài toán.

---

## 4.2. Questions

`questions` là các câu hỏi annotation mà annotator cần trả lời.

Trong ví dụ này có hai question:

### Text classification

Dùng `LabelQuestion`.

```python
rg.LabelQuestion(
    name="label",
    title="Classify the text:",
    labels=data.unique("label_text")
)
```

Ý nghĩa:

- `name="label"`: tên nội bộ của question.
- `title="Classify the text:"`: câu hỏi hiển thị trên UI.
- `labels=data.unique("label_text")`: danh sách nhãn lấy từ dataset gốc.

`LabelQuestion` phù hợp khi cần chọn một nhãn cho toàn bộ record.

Ví dụ task:

```text
Bài báo này thuộc chủ đề nào?
- World
- Sports
- Business
- Sci/Tech
```

---

### Token classification / NER

Dùng `SpanQuestion`.

```python
rg.SpanQuestion(
    name="entities",
    title="Highlight all the entities in the text:",
    labels=["PERSON", "ORG", "LOC", "EVENT"],
    field="text",
)
```

Ý nghĩa:

- Annotator sẽ bôi đen một đoạn text.
- Sau đó gán label cho đoạn đó.
- `field="text"` cho biết span được annotate trên field nào.

`SpanQuestion` là lựa chọn phù hợp cho token classification hoặc named entity recognition.

Ví dụ:

```text
"Apple announced a new product in California."

Apple -> ORG
California -> LOC
```

---

## 4.3. Full settings example

```python
settings = rg.Settings(
    fields=[
        rg.TextField(name="text")
    ],
    questions=[
        rg.LabelQuestion(
            name="label",
            title="Classify the text:",
            labels=data.unique("label_text"),
        ),
        rg.SpanQuestion(
            name="entities",
            title="Highlight all the entities in the text:",
            labels=["PERSON", "ORG", "LOC", "EVENT"],
            field="text",
        ),
    ],
)
```

---

# 5. Tạo dataset trong Argilla

Sau khi có settings:

```python
dataset = rg.Dataset(
    name="ag_news",
    settings=settings,
)

dataset.create()
```

Lúc này dataset đã xuất hiện trong Argilla UI, nhưng chưa có records.

---

# 6. Upload records vào Argilla

Dùng:

```python
dataset.records.log(
    data,
    mapping={"label_text": "label"}
)
```

Điểm kỹ thuật quan trọng nằm ở `mapping`.

Dataset gốc có cột:

```text
label_text
```

Trong Argilla question lại tên là:

```text
label
```

Vì vậy cần map:

```python
mapping={"label_text": "label"}
```

Điều này giúp Argilla dùng `label_text` làm **suggested label** cho question `label`.

Tức là khi annotator mở UI, text classification đã có nhãn gợi ý sẵn. Annotator chỉ cần rà soát, sửa nếu sai, rồi submit.

---

# 7. Annotation trong Argilla UI

Sau khi upload records, annotator có thể vào UI để gán nhãn.

## 7.1. Viết annotation guidelines

Trước khi annotate, đặc biệt khi làm việc nhóm, nên viết rõ:

- Mỗi label nghĩa là gì.
- Khi nào dùng label nào.
- Trường hợp nhập nhằng thì xử lý ra sao.
- Ví dụ đúng/sai.
- Quy tắc thống nhất cho span annotation.

Ví dụ với NER:

```text
PERSON: tên người thật hoặc nhân vật cụ thể.
ORG: tổ chức, công ty, đội nhóm, cơ quan.
LOC: địa điểm địa lý.
EVENT: sự kiện có tên riêng.
```

Guidelines giúp giảm mâu thuẫn giữa annotators và tăng chất lượng dataset.

---

## 7.2. Distribution settings

Argilla có thể cấu hình số lượng response tối thiểu cho mỗi record.

Mặc định:

```text
minimum submitted responses = 1
```

Nghĩa là chỉ cần một annotator submit, record được coi là completed.

Nếu muốn đo **inter-annotator agreement**, có thể tăng giá trị này lên, ví dụ:

```text
minimum submitted responses = 3
```

Khi đó mỗi record cần ít nhất 3 submitted responses mới được coi là completed.

Lưu ý: số này phải nhỏ hơn hoặc bằng tổng số annotators.

---

## 7.3. Các trạng thái khi annotate

Annotator có thể:

### Submit

Hoàn thành response cho record.

```text
Record được tính vào tiến độ nếu đủ số submitted responses tối thiểu.
```

### Save as draft

Lưu nháp, chưa submit.

Dùng khi annotator chưa chắc chắn hoặc muốn quay lại sau.

### Discard

Loại record khỏi quá trình annotation.

Dùng nếu record lỗi, không liên quan, bị trùng, hoặc không thể annotate đáng tin cậy.

---

# 8. Dùng dataset đã annotate

Sau khi annotation xong, bạn có thể dùng lại dataset trong Python.

## 8.1. Kết nối lại Argilla

```python
import argilla as rg

client = rg.Argilla(
    api_url="https://<your-username>.<space-name>.hf.space",
    api_key="...",
)
```

Nếu Space private:

```python
client = rg.Argilla(
    api_url="https://<your-username>.<space-name>.hf.space",
    api_key="...",
    headers={"Authorization": f"Bearer {HF_TOKEN}"},
)
```

---

## 8.2. Load dataset từ Argilla server

```python
dataset = client.datasets(name="ag_news")
```

Sau đó có thể truy cập records:

```python
records = dataset.records
```

---

# 9. Filter records theo trạng thái

Thường bạn chỉ muốn dùng các records đã hoàn thành.

Tạo filter:

```python
status_filter = rg.Query(
    filter=rg.Filter([
        ("status", "==", "completed")
    ])
)
```

Lấy records đã completed:

```python
filtered_records = dataset.records(status_filter)
```

Quan trọng: `completed` nghĩa là record đã đạt đủ số submitted responses tối thiểu theo distribution settings.

Một record completed vẫn có thể có nhiều response, và từng response có thể ở trạng thái:

- `submitted`
- `draft`
- `discarded`

Vì vậy nếu cần training dataset sạch, bạn nên kiểm tra kỹ logic chọn response cuối cùng hoặc response được chấp nhận.

---

# 10. Export dataset sang Hugging Face Hub

Có hai cách chính.

---

## 10.1. Export subset records sang `datasets.Dataset`

Nếu bạn đã filter records, ví dụ chỉ lấy completed records:

```python
filtered_records.to_datasets().push_to_hub(
    "argilla/ag_news_annotated"
)
```

Luồng xử lý:

```text
Argilla records -> Hugging Face Dataset -> push_to_hub
```

Cách này phù hợp nếu bạn muốn chỉ export phần dữ liệu đã lọc, ví dụ:

- Chỉ records completed.
- Chỉ records có nhãn nhất định.
- Chỉ records không bị discard.
- Chỉ annotations từ annotator cụ thể.

---

## 10.2. Export toàn bộ Argilla dataset

```python
dataset.to_hub(
    repo_id="argilla/ag_news_annotated"
)
```

Cách này export toàn bộ dataset, bao gồm cả settings của Argilla.

Điểm mạnh: người khác có thể import lại dataset vào Argilla với đầy đủ cấu hình task.

```python
dataset = rg.Dataset.from_hub(
    repo_id="argilla/ag_news_annotated"
)
```

Cách này phù hợp nếu muốn chia sẻ dataset như một Argilla project hoàn chỉnh.

---

# 11. Các khái niệm kỹ thuật quan trọng cần nắm

## Argilla instance

Server chạy Argilla UI và API.

Có thể chạy trên:

- Hugging Face Spaces
- Local Docker
- Deployment riêng

---

## Argilla Python SDK

Thư viện Python dùng để:

- Kết nối server.
- Tạo dataset.
- Định nghĩa settings.
- Upload records.
- Query/filter records.
- Export dataset.

Cài bằng:

```python
pip install argilla
```

---

## Field

Dữ liệu đầu vào mà annotator nhìn thấy.

Ví dụ:

```python
rg.TextField(name="text")
```

Một dataset có thể có nhiều fields, không bắt buộc chỉ một field.

---

## Question

Tác vụ annotation cần annotator thực hiện.

Một số loại thường gặp:

```python
rg.LabelQuestion      # chọn một label cho cả record
rg.SpanQuestion       # đánh dấu span trong text
rg.TextQuestion       # nhập text tự do
```

Trong chương này:

- Text classification dùng `LabelQuestion`.
- Token classification dùng `SpanQuestion`.

---

## Suggestion / pre-annotation

Nhãn có sẵn được đưa vào Argilla để annotator rà soát.

Ví dụ:

```python
dataset.records.log(
    data,
    mapping={"label_text": "label"}
)
```

Ở đây `label_text` từ dataset gốc trở thành nhãn gợi ý cho question `label`.

Argilla không tự động tạo suggested labels nếu bạn không cung cấp.

---

## Completed record

Record được xem là completed nếu đạt đủ số lượng submitted responses tối thiểu.

Ví dụ:

```text
minimum submitted responses = 1
```

Chỉ cần một annotator submit là completed.

---

# 12. Workflow đầy đủ

Một pipeline điển hình với Argilla:

```python
import argilla as rg
from datasets import load_dataset

# 1. Connect to Argilla
client = rg.Argilla(
    api_url="https://<your-username>.<space-name>.hf.space",
    api_key="...",
)

# 2. Load source dataset
data = load_dataset("SetFit/ag_news", split="train")

# 3. Define Argilla settings
settings = rg.Settings(
    fields=[
        rg.TextField(name="text")
    ],
    questions=[
        rg.LabelQuestion(
            name="label",
            title="Classify the text:",
            labels=data.unique("label_text"),
        ),
        rg.SpanQuestion(
            name="entities",
            title="Highlight all the entities in the text:",
            labels=["PERSON", "ORG", "LOC", "EVENT"],
            field="text",
        ),
    ],
)

# 4. Create dataset
dataset = rg.Dataset(
    name="ag_news",
    settings=settings,
)

dataset.create()

# 5. Upload records with pre-annotations
dataset.records.log(
    data,
    mapping={"label_text": "label"}
)

# 6. Later: load dataset
dataset = client.datasets(name="ag_news")

# 7. Filter completed records
status_filter = rg.Query(
    filter=rg.Filter([
        ("status", "==", "completed")
    ])
)

filtered_records = dataset.records(status_filter)

# 8. Export filtered records to Hub
filtered_records.to_datasets().push_to_hub(
    "your-username/ag_news_annotated"
)
```

---

# 13. Những điểm dễ nhầm

## Argilla không phải công cụ train model

Sai:

```text
Dùng Argilla để train model.
```

Đúng:

```text
Dùng Argilla để tạo dataset chất lượng, rồi dùng dataset đó để train model.
```

---

## Không bắt buộc dùng Hugging Face Spaces

Argilla có thể chạy local bằng Docker hoặc deployment riêng.

Hugging Face Spaces chỉ là cách dễ nhất trong khóa học.

---

## Không phải lúc nào cũng cần HF token

Chỉ cần HF token nếu Argilla Space là private.

Nếu Space public hoặc chạy local, thường không cần.

---

## Field khác metadata

- `fields`: nội dung annotator cần xem và annotate.
- `metadata`: thông tin phụ để filter/sort, không phải nội dung chính để annotate.

---

## Token classification nên dùng `SpanQuestion`

Vì task này cần đánh dấu một đoạn text cụ thể.

Không nên dùng `LabelQuestion` cho token classification, vì `LabelQuestion` chỉ gán nhãn cho toàn record.

---

# 14. Kết luận

Chapter 10 hướng dẫn một workflow rất thực tế:

1. Deploy Argilla.
2. Kết nối bằng Python SDK.
3. Load dataset từ Hugging Face Hub.
4. Định nghĩa fields và questions.
5. Upload records vào Argilla.
6. Annotate bằng UI.
7. Filter records đã hoàn thành.
8. Export dataset lên Hugging Face Hub.

Điểm cốt lõi cần nhớ: **chất lượng model phụ thuộc rất nhiều vào chất lượng dataset**. Argilla là công cụ giúp bạn cải thiện dataset một cách có hệ thống, có UI, có quy trình làm việc nhóm, có khả năng lọc/export và tích hợp tốt với Hugging Face ecosystem.