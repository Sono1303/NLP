# Lab 6: Introduction to Transformers

## 1. Giới thiệu

Lab này giới thiệu về kiến trúc Transformer và cách sử dụng thư viện Transformers của Hugging Face để thực hiện các tác vụ NLP cơ bản.

## 1.1. Dataset sử dụng

**Tên dataset**: Demo sentences (Synthetic data)

**Mô tả**: Lab này sử dụng các câu demo được tạo thủ công để minh họa khả năng của các mô hình Transformer pre-trained.

**Cấu trúc dữ liệu**:
- **Masked Language Modeling**: Câu có token [MASK] cần điền
- **Text Generation**: Câu mồi để sinh tiếp văn bản
- **Sentence Embedding**: Câu mẫu để tính vector biểu diễn
- **Models sử dụng**: 
  - BERT (bert-base-uncased) - Encoder-only
  - GPT-2 (gpt2) - Decoder-only

**Lưu ý**: Không cần tải dataset external, tất cả được tạo trong code demo.

## 2. Cài đặt thư viện cần thiết

```python
!pip install transformers torch
```

## 3. Bài tập thực hành

### Bài 1: Khôi phục Masked Token (Masked Language Modeling)

Sử dụng pipeline `fill-mask` để dự đoán từ bị che trong câu: `Hanoi is the [MASK] of Vietnam.`

```python
from transformers import pipeline

# Chỉ định framework PyTorch để tránh lỗi TensorFlow/Keras
mask_filler = pipeline("fill-mask", model="bert-base-uncased", framework="pt")
input_sentence = 'Hanoi is the [MASK] of Vietnam.'
predictions = mask_filler(input_sentence, top_k=5)

print(f'Câu gốc: {input_sentence}')
for pred in predictions:
    print(f"Dự đoán: '{pred['token_str']}' với độ tin cậy: {pred['score']:.4f}")
    print(f" -> Câu hoàn chỉnh: {pred['sequence']}")
```

**Câu hỏi:**
1. Mô hình đã dự đoán đúng từ 'capital' không?
2. Tại sao các mô hình Encoder-only như BERT lại phù hợp cho tác vụ này?

**Trả lời:**
1. **Có**, mô hình đã dự đoán đúng từ 'capital' với độ tin cậy rất cao (99.91%). Kết quả top-5:
   - capital: 99.91%
   - center: 0.01%
   - birthplace: 0.01%
   - headquarters: 0.01%
   - city: 0.01%

2. BERT được huấn luyện với nhiệm vụ Masked Language Modeling, cho phép mô hình nhìn cả trái và phải của token bị che để dự đoán chính xác từ bị thiếu. Cơ chế **bidirectional** (hai chiều) giúp BERT hiểu ngữ cảnh đầy đủ: "Hanoi is the" (bên trái) và "of Vietnam" (bên phải), từ đó suy luận chính xác từ thiếu là "capital".

### Bài 2: Dự đoán từ tiếp theo (Next Token Prediction)

Sử dụng pipeline `text-generation` để sinh tiếp cho câu: `The best thing about learning NLP is`

```python
from transformers import pipeline

# Chỉ định framework PyTorch để tránh lỗi TensorFlow
generator = pipeline("text-generation", model="gpt2", framework="pt")
prompt = "The best thing about learning NLP is"
generated_texts = generator(prompt, max_length=50, num_return_sequences=1)

print(f"Câu mồi: '{prompt}'")
for text in generated_texts:
    print("Văn bản được sinh ra:")
    print(text['generated_text'])
```

**Câu hỏi:**
1. Kết quả sinh ra có hợp lý không?
2. Tại sao các mô hình Decoder-only như GPT lại phù hợp cho tác vụ này?

**Trả lời:**
1. **Có một phần hợp lý**. Văn bản được sinh ra:
   > "The best thing about learning NLP is that you always get to take what they have said and put it right there in a way that you feel like you understand."
   
   Câu này có cấu trúc ngữ pháp đúng và ý nghĩa liên quan đến việc hiểu ngôn ngữ. Tuy nhiên, nội dung hơi chung chung và phần cuối bị cắt đột ngột do giới hạn `max_length=50`. Đây là hạn chế của mô hình GPT-2 nhỏ và cần điều chỉnh tham số để có kết quả tốt hơn.

2. GPT được huấn luyện với nhiệm vụ **dự đoán từ tiếp theo** (next token prediction) dựa trên chuỗi đã có. Kiến trúc **unidirectional** (một chiều, từ trái sang phải) phù hợp cho sinh văn bản tự nhiên vì mô hình chỉ cần xem các từ trước đó để dự đoán từ kế tiếp, tương tự cách con người viết văn bản tuần tự.

### Bài 3: Tính toán vector biểu diễn của câu (Sentence Representation)

Tính vector biểu diễn cho câu `This is a sample sentence.` bằng phương pháp Mean Pooling với BERT.

```python
import torch
from transformers import AutoTokenizer, AutoModel

model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

sentences = ['This is a sample sentence.']
inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

with torch.no_grad():
    outputs = model(**inputs)

last_hidden_state = outputs.last_hidden_state
attention_mask = inputs['attention_mask']

mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
sentence_embedding = sum_embeddings / sum_mask

print('Vector biểu diễn của câu:')
print(sentence_embedding)
print('Kích thước của vector:', sentence_embedding.shape)
```

**Câu hỏi:**
1. Kích thước (chiều) của vector biểu diễn là bao nhiêu? Con số này tương ứng với tham số nào của mô hình BERT?
2. Tại sao chúng ta cần sử dụng `attention_mask` khi thực hiện Mean Pooling?

**Trả lời:**
1. **Kích thước: torch.Size([1, 768])**
   - **1**: batch size (1 câu)
   - **768**: chiều của hidden state
   
   Con số **768** tương ứng với tham số `hidden_size` của mô hình **bert-base-uncased**. Đây là số chiều của vector đầu ra từ mỗi lớp Transformer trong BERT. Các biến thể BERT khác có hidden_size khác nhau (ví dụ: bert-large có 1024).

2. `attention_mask` giúp loại bỏ ảnh hưởng của các **token padding** khi tính trung bình. Khi xử lý nhiều câu cùng lúc (batch), các câu ngắn sẽ được thêm padding để cùng độ dài. Nếu không dùng attention_mask, các token padding sẽ được tính vào trung bình, làm vector biểu diễn không chính xác. Attention_mask đảm bảo chỉ các token thực sự của câu được tính, cho kết quả đúng ngữ nghĩa.

## 4. Kết luận

Trong lab này, chúng ta đã:
- Tìm hiểu về kiến trúc Transformer với 3 loại chính: Encoder-only (BERT), Decoder-only (GPT), và Encoder-Decoder
- Thực hành với các pipeline của Hugging Face cho Masked Language Modeling và Text Generation
- Hiểu cách tính toán sentence embeddings với Mean Pooling
- So sánh ưu điểm của từng kiến trúc cho các tác vụ NLP khác nhau

## 5. Tài liệu tham khảo

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)
- [BERT Paper: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [GPT-2 Paper: Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
