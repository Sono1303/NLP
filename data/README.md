# Datasets - Metadata và Mô tả

**LƯU Ý QUAN TRỌNG**: Thư mục này KHÔNG chứa datasets thực tế. Chỉ chứa metadata, mô tả cấu trúc, và hướng dẫn tải datasets.

## Tổng quan

Repository này sử dụng nhiều datasets công khai cho các bài lab NLP. Do kích thước lớn, datasets không được push lên GitHub mà chỉ được lưu local.

## Danh sách Datasets

### 1. Universal Dependencies English-EWT (UD_English-EWT)

**Sử dụng trong**: Lab 1, Lab 3, Lab 5 Part 3

**Mô tả**: Corpus tiếng Anh được gán nhãn ngữ pháp theo chuẩn Universal Dependencies

**Cấu trúc**:
- Format: CoNLL-U
- Train: ~12,544 câu
- Dev: ~2,001 câu  
- Test: ~2,077 câu
- Các cột: ID, FORM, LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL, DEPS, MISC

**Nguồn**: 
- GitHub: https://github.com/UniversalDependencies/UD_English-EWT
- Website: https://universaldependencies.org/

**Cách tải**:
```bash
cd Lab1/UD_English-EWT
# hoặc
cd Lab3/data/UD_English-EWT

# Tải từ GitHub
git clone https://github.com/UniversalDependencies/UD_English-EWT.git
```

**Files chính**:
- `en_ewt-ud-train.conllu`: Training set
- `en_ewt-ud-dev.conllu`: Development/validation set
- `en_ewt-ud-test.conllu`: Test set

---

### 2. GloVe Pre-trained Vectors

**Sử dụng trong**: Lab 3

**Mô tả**: Pre-trained word embeddings từ Stanford NLP

**Cấu trúc**:
- Format: Text file (word + vector values)
- Vocabulary: 400,000 words
- Dimensions: 50d, 100d, 200d, 300d

**Nguồn**: 
- Website: https://nlp.stanford.edu/projects/glove/
- Direct: http://nlp.stanford.edu/data/glove.6B.zip

**Cách tải**:
```bash
cd Lab3/data
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip -d glove.6B/
```

**Files**:
- `glove.6B.50d.txt`: 50 dimensions
- `glove.6B.100d.txt`: 100 dimensions (used in lab)
- `glove.6B.200d.txt`: 200 dimensions
- `glove.6B.300d.txt`: 300 dimensions

---

### 3. C4 Dataset (Subset)

**Sử dụng trong**: Lab 3

**Mô tả**: Subset của Colossal Clean Crawled Corpus (C4)

**Cấu trúc**:
- Format: JSON
- Size: ~30K documents (subset)
- Fields: text, url, timestamp

**Nguồn**: 
- Hugging Face: https://huggingface.co/datasets/c4

**Cách tải**:
```bash
cd Lab3/data
# Download subset qua Hugging Face datasets library
```

**File**: `c4-train.00000-of-01024-30K.json`

---

### 4. HWU-64 Dataset (Intent Classification)

**Sử dụng trong**: Lab 5 Part 2

**Mô tả**: Dataset cho bài toán Intent Classification với 64 intents

**Cấu trúc**:
- Format: CSV
- Fields: text, label
- Train: ~8,954 samples
- Dev: ~1,076 samples
- Test: ~1,076 samples
- Number of classes: 64 intents

**Nguồn**: 
- GitHub: https://github.com/xliuhw/NLU-Evaluation-Data

**Cách tải**:
```bash
cd Lab5/data/hwu
# Download từ GitHub repository
```

**Files**:
- `train.csv`: Training set
- `val.csv`: Validation set
- `test.csv`: Test set

**Các intent classes**: alarm_query, alarm_remove, alarm_set, audio_volume_down, calendar_query, iot_hue_lightoff, music_query, play_music, qa_factoid, weather_query, ...

---

### 5. CoNLL-2003 NER Dataset

**Sử dụng trong**: Lab 5 Part 4

**Mô tả**: Dataset chuẩn cho Named Entity Recognition

**Cấu trúc**:
- Format: CoNLL format (token + POS + chunk + NER)
- Train: ~14,041 câu
- Dev: ~3,250 câu
- Test: ~3,453 câu
- Entity types: PER (Person), LOC (Location), ORG (Organization), MISC (Miscellaneous)
- Tagging scheme: IOB (Inside, Outside, Beginning)

**Nguồn**: 
- Website: https://www.clips.uantwerpen.be/conll2003/ner/
- Hugging Face: https://huggingface.co/datasets/conll2003

**Cách tải**:
```bash
cd Lab5/part4/data/conll2003
# Sử dụng Hugging Face datasets library
```

**Files**:
- `train.txt`: Training set
- `valid.txt`: Validation set
- `test.txt`: Test set

**Format mỗi dòng**: `token POS-tag chunk-tag NER-tag`

---

### 6. Sentiment Analysis Datasets

**Sử dụng trong**: Lab 4

**Mô tả**: Các datasets cho phân tích cảm xúc

**Cấu trúc**:
- Format: CSV
- Fields: text, sentiment/label
- Labels: -1 (negative), 1 (positive) hoặc 0, 1, 2 (negative, neutral, positive)

**Nguồn**: 
- Twitter Financial News: https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment
- Kaggle sentiment datasets

**Files**:
- `sentiments.csv`: General sentiment dataset
- `twitter-financial-news-sentiment/`: Financial domain

---

## Cấu trúc Thư mục Data trong các Lab

```
Lab1/
  └── UD_English-EWT/          # Universal Dependencies dataset

Lab3/
  └── data/
      ├── glove.6B/             # GloVe vectors
      ├── c4-train-*.json       # C4 subset
      └── UD_English-EWT/       # For Word2Vec training

Lab4/
  └── data/
      ├── sentiments.csv
      └── twitter-financial-news-sentiment/

Lab5/
  ├── data/
  │   ├── hwu/                  # Intent classification
  │   └── nlu.csv/
  └── part4/data/
      └── conll2003/            # NER dataset
```

## Lưu ý về .gitignore

Các file sau đã được thêm vào `.gitignore`:
- `*.csv` (except metadata)
- `*.json` (except config)
- `*.txt` (except README)
- `*.conllu`
- `*.vec`
- `*.bin`
- Model files: `*.pt`, `*.pth`, `*.model`

## Hướng dẫn Tải tất cả Datasets

Tạo script để tải tự động:

```bash
# Tạo file download_datasets.sh
#!/bin/bash

# Lab1 - UD English-EWT
cd Lab1
git clone https://github.com/UniversalDependencies/UD_English-EWT.git

# Lab3 - GloVe
cd ../Lab3/data
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip -d glove.6B/

# Lab5 - Datasets qua Hugging Face
pip install datasets
python -c "
from datasets import load_dataset
# HWU-64
hwu = load_dataset('SetFit/hwu_64')
# CoNLL-2003
conll = load_dataset('conll2003')
"

echo "All datasets downloaded!"
```

## Kích thước Ước tính

| Dataset | Size | Files |
|---------|------|-------|
| UD English-EWT | ~20 MB | 3 files |
| GloVe 6B | ~822 MB | 4 files |
| C4 subset | ~100 MB | 1 file |
| HWU-64 | ~2 MB | 3 files |
| CoNLL-2003 | ~5 MB | 3 files |
| Sentiment datasets | ~50 MB | Multiple |
| **Total** | **~1 GB** | |

## Trích dẫn

Nếu sử dụng datasets này trong nghiên cứu, vui lòng trích dẫn:

**Universal Dependencies**:
```
@inproceedings{nivre2016universal,
  title={Universal Dependencies v1: A Multilingual Treebank Collection},
  author={Nivre, Joakim and others},
  booktitle={LREC},
  year={2016}
}
```

**GloVe**:
```
@inproceedings{pennington2014glove,
  title={Glove: Global vectors for word representation},
  author={Pennington, Jeffrey and Socher, Richard and Manning, Christopher D},
  booktitle={EMNLP},
  year={2014}
}
```

**CoNLL-2003**:
```
@inproceedings{tjong2003introduction,
  title={Introduction to the CoNLL-2003 shared task: Language-independent named entity recognition},
  author={Tjong Kim Sang, Erik F and De Meulder, Fien},
  booktitle={CoNLL},
  year={2003}
}
```
