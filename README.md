
# Natural Language Processing (NLP) - Lab Exercises

Repository này chứa các bài thực hành môn Xử lý Ngôn ngữ Tự nhiên (NLP), bao gồm các kỹ thuật cơ bản đến nâng cao trong lĩnh vực NLP.

## Cấu trúc Repository

```
./
├── src/                # Source code Python
├── spark_labs/         # Source code Scala
├── notebook/           # Jupyter notebooks cho các bài lab
├── test/               # Test scripts và unit tests
├── data/               # Metadata và mô tả datasets (KHÔNG chứa dataset thực tế)
├── report/             # Báo cáo chi tiết cho từng lab
├── Lab1-Lab6/          # Thư mục gốc chứa code, data, results của từng lab
├── README.md           # File này
├── .gitignore          # Loại bỏ files không cần thiết
└── more.md             # Hướng dẫn và yêu cầu bổ sung
```

## Danh sách Labs

### Lab 1: Preprocessing và Dataset Loading
- **Mô tả**: Xử lý dữ liệu văn bản, tokenization, stemming, lemmatization
- **Dataset**: Universal Dependencies English-EWT
- **Báo cáo**: [Lab1_report.md](report/Lab1_report.md)
- **Code**: [Lab1/](Lab1/)

### Lab 2: Spark MLlib và Xử lý Dữ liệu Lớn
- **Mô tả**: Sử dụng Apache Spark cho xử lý NLP trên dữ liệu lớn
- **Technologies**: Scala, Spark MLlib
- **Báo cáo**: [Lab2_report.md](report/Lab2_report.md)
- **Code**: [Lab2/spark_labs/](Lab2/spark_labs/)

### Lab 3: Word Embeddings
- **Phần 1**: Trực quan hóa và phân tích embedding
  - Báo cáo: [Lab3_part1_report.md](report/Lab3_part1_report.md)
- **Phần 2**: Word2Vec và training embeddings
  - Báo cáo: [Lab3_part2_report.md](report/Lab3_part2_report.md)
- **Datasets**: GloVe, UD English-EWT, C4
- **Code**: [Lab3/](Lab3/)

### Lab 4: Text Classification & Sentiment Analysis
- **Mô tả**: Phân loại văn bản và phân tích cảm xúc với Spark MLlib
- **Models**: Logistic Regression, Naive Bayes, GBT, Neural Networks
- **Báo cáo**: [Lab4_report.md](report/Lab4_report.md)
- **Code**: [Lab4/](Lab4/)

### Lab 5: RNN và Deep Learning cho NLP
- **Phần 1**: PyTorch Introduction
  - Báo cáo: [Lab5_part1_report.md](report/Lab5_part1_report.md)
- **Phần 2**: RNN cho Text Classification
  - Báo cáo: [Lab5_part2_report.md](report/Lab5_part2_report.md)
- **Phần 3**: RNN cho POS Tagging
  - Báo cáo: [Lab5_part3_report.md](report/Lab5_part3_report.md)
- **Phần 4**: RNN cho Named Entity Recognition (NER)
  - Báo cáo: [Lab5_part4_report.md](report/Lab5_part4_report.md)
- **Datasets**: HWU-64, UD English-EWT, CoNLL-2003
- **Code**: [Lab5/](Lab5/)

### Lab 6: Transformers và Attention Mechanism
- **Mô tả**: Giới thiệu về Transformer architecture và attention
- **Báo cáo**: [Lab6/part1/](Lab6/part1/)
- **Code**: [Lab6/](Lab6/)

## Technologies & Libraries

- **Python**: PyTorch, Gensim, scikit-learn, pandas, numpy, matplotlib
- **Scala**: Apache Spark MLlib
- **Deep Learning**: RNN, LSTM, Transformers
- **NLP Tools**: Word2Vec, GloVe, TF-IDF

## Datasets

**Lưu ý**: Repository này KHÔNG chứa datasets thực tế để tránh quá tải. Các datasets được mô tả trong [data/README.md](data/README.md).

Các datasets sử dụng:
- Universal Dependencies English-EWT (UD_English-EWT)
- GloVe pre-trained vectors
- C4 dataset (subset)
- HWU-64 (Intent Classification)
- CoNLL-2003 (Named Entity Recognition)
- Sentiment datasets

## Getting Started

### Yêu cầu hệ thống
```bash
# Python 3.8+
pip install -r requirements.txt

# Apache Spark (for Lab 2, 4)
# Scala 2.12+
```

### Chạy các Lab

**Lab 1-3** (Python):
```bash
cd Lab1  # hoặc Lab3
python test/main.py
```

**Lab 2** (Spark/Scala):
```bash
cd Lab2/spark_labs
sbt run
```

**Lab 4-5** (PyTorch):
```bash
cd Lab4  # hoặc Lab5
python test/lab5_test.py
jupyter notebook  # Cho các notebooks
```

## Báo cáo

Tất cả báo cáo chi tiết cho từng lab được lưu trong thư mục [report/](report/), bao gồm:
- Giải thích các bước thực hiện
- Hướng dẫn chạy code
- Phân tích kết quả và đánh giá
- Khó khăn gặp phải và giải pháp
- Tài liệu tham khảo

## Lưu ý quan trọng

1. **Datasets**: Không push datasets lớn lên GitHub. Sử dụng `.gitignore` để loại bỏ.
2. **Models**: Model files (`.pt`, `.pth`) được ignore, chỉ giữ lại trained models quan trọng.
3. **Reports**: Ưu tiên format markdown (.md) để dễ đọc và trích xuất thông tin.
4. **Code organization**: Code được tổ chức theo từng lab, với structure rõ ràng.

## Tài liệu tham khảo

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Apache Spark MLlib](https://spark.apache.org/docs/latest/ml-guide.html)
- [Universal Dependencies](https://universaldependencies.org/)
- [GloVe: Global Vectors](https://nlp.stanford.edu/projects/glove/)
- [CoNLL-2003 NER Dataset](https://www.clips.uantwerpen.be/conll2003/ner/)

## Tác giả

- **Sono1303**
- Repository: [NLP](https://github.com/Sono1303/NLP)

## Cập nhật

- Tuần 12/2025: Tái cấu trúc repository theo chuẩn, hoàn thiện báo cáo Lab 1-6
- Tuần 11/2025: Hoàn thành Lab 5 (RNN/LSTM)
- Tuần 10/2025: Lab 4 (Text Classification & Sentiment Analysis)

---

**Lưu ý**: Repository này được tổ chức theo yêu cầu của môn học. Xem [more.md](more.md) để biết thêm chi tiết về cấu trúc và yêu cầu.