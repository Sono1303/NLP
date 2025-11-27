
# Lab 3: Word Embeddings

## Phần 2: Word Embeddings với Word2Vec

### 1. Source code, dữ liệu, kết quả sử dụng
- [Lab3/test/lab4_test.py](../Lab3/test/lab4_test.py): Test pre-trained GloVe
- [Lab3/test/lab4_embedding_training_demo.py](../Lab3/test/lab4_embedding_training_demo.py): Custom Word2Vec training
- [Lab3/test/lab4_spark_word2vec_demo.py](../Lab3/test/lab4_spark_word2vec_demo.py): Spark MLlib training
- [Lab3/results/](../Lab3/results/): Kết quả test, training
- [Lab3/data/](../Lab3/data/): Dữ liệu GloVe, UD English-EWT, C4

### 1.1. Datasets sử dụng

**Dataset 1**: GloVe Pre-trained Vectors (giống Lab 3 Part 1)
- **File**: `glove.6B.300d.txt`, `wiki-news-300d-1M-subword.vec`
- **Kích thước vector**: 300d
- Xem [data/README.md](../data/README.md) để biết chi tiết

**Dataset 2**: UD English-EWT (training Word2Vec)
- **Format**: CoNLL-U
- **Số lượng**: ~12,544 câu (train set)
- Sử dụng cho custom Word2Vec training

**Dataset 3**: C4 Subset (Spark Word2Vec)
- **Format**: JSON
- **Số lượng**: 30,000 documents
- Sử dụng cho Spark MLlib Word2Vec training

**Lưu ý**: Tất cả datasets không được commit lên GitHub. Xem [data/README.md](../data/README.md).

### 2. Yêu cầu hệ thống
- Python 3.8+
- Jupyter Notebook
- Apache Spark (cho phần Spark MLlib)

### 3. Cài đặt dependencies
```bash
pip install gensim numpy matplotlib scikit-learn pandas
pip install pyspark  # Cho phần Spark
```

### 4. Chạy các thành phần

#### 4.1 Pre-trained Model Test
```bash
cd Lab3
python test/lab4_test.py
```
**Output**: `results/lab4_test_output.txt`

#### 4.2 Custom Word2Vec Training
```bash
python test/lab4_embedding_training_demo.py
```
**Output**: `results/lab4_training_demo_output.txt`

#### 4.3 Spark MLlib Training
```bash
python test/lab4_spark_word2vec_demo.py
```
**Output**: `results/lab4_spark_word2vec_output.txt`

#### 4.4 Visualization (Jupyter Notebook)
```bash
jupyter notebook Lab3.ipynb
```
Chạy tất cả cells để xem trực quan hóa PCA và phân tích word clusters.

### 5. Phân tích kết quả chi tiết

#### 5.1 Pre-trained Model (GloVe) Analysis

##### Kết quả chính
- **Model**: GloVe Wiki Gigaword 50D với 400,000 từ
- **Vector quality**: Vectors có giá trị thực tế, không phải zero vectors
- **Similarity scores**: 
  - king-queen: 0.7839 (rất cao, thể hiện mối quan hệ gender)
  - king-man: 0.5309 (vừa phải, thể hiện mối quan hệ hierarchical)

#### Phân tích từ đồng nghĩa cho "computer"
```
1. computers (0.917)  - Dạng số nhiều, hoàn hảo
2. software (0.881)   - Khái niệm liên quan, hợp lý
3. technology (0.853) - Phạm vi rộng hơn, hợp lý
4. electronic (0.813) - Mối quan hệ phần cứng
5. internet (0.806)   - Bối cảnh sử dụng
```

**Nhận xét**: Pre-trained model thể hiện khả năng nắm bắt:
- **Mối quan hệ hình thái từ**: computer ↔ computers
- **Trường ngữ nghĩa**: computer ↔ software, technology
- **Liên kết khái niệm**: computer ↔ electronic, internet

#### 5.2 Custom Word2Vec Training Analysis

##### Kết quả training
- **Dataset**: 13,572 sentences từ UD English-EWT
- **Training time**: 1.58 seconds
- **Vocabulary**: 3,772 words
- **Vector dimensions**: 50D

##### Chất lượng học được
```
Similarities:
- the-man: 0.575
- man-woman: 0.820
```

**Phân tích**: 
- Model tự train đạt similarity scores hợp lý
- Mối quan hệ gender (man-woman: 0.820) được học tốt
- Limited vocabulary do dataset nhỏ, nhưng chất lượng ở mức chấp nhận được

#### 5.3 Spark MLlib Large Dataset Analysis

##### Kết quả ấn tượng
- **Dataset**: 29,971 documents từ C4
- **Thời gian training**: 5.85 phút
- **Vocabulary**: 78,930 từ (rất lớn!)
- **Chiều vector**: 100D

##### So sánh chất lượng
```
Tương tự với 'computer':
1. computers (0.798)
2. desktop (0.702)
3. laptop (0.680)
4. software (0.672)
```

**Phân tích**:
- Vocabulary 20x lớn hơn custom model
- Mối quan hệ ngữ nghĩa rõ ràng: computer → desktop, laptop
- Training trên dataset lớn cho kết quả robust hơn

#### 5.4 Visualization Analysis (PCA + Scatter Plot)

##### Phương pháp
- **Giảm chiều**: PCA từ 100D → 2D
- **Phương sai giải thích**: ~10-15% (đánh đổi phù hợp cho visualization)
- **Trực quan hóa**: Scatter plot với vector arrows từ gốc tọa độ

##### Phân tích biểu đồ trực quan hóa

###### Quan sát clustering cho từ "king"
```
Các từ tương tự được nhóm: queen, prince, kingdom, royal...
```

**Kết quả quan sát**:
1. **Phân cụm không gian**: Các từ có liên quan về mặt ngữ nghĩa có xu hướng gần nhau trong không gian 2D
2. **Mối quan hệ giới tính**: "king" và "queen" có khoảng cách hợp lý, thể hiện khái niệm tương tự nhưng khác giới tính
3. **Mối quan hệ cấp bậc**: "prince", "duke" cluster gần "king", thể hiện thứ bậc hoàng gia
4. **Mối quan hệ bối cảnh**: "kingdom", "castle" gần nhau, thể hiện bối cảnh lĩnh vực

###### Cụm từ thú vị được phát hiện
- **Cụm hoàng gia**: king, queen, prince, royal, kingdom
- **Cụm công nghệ**: computer, software, technology, electronic
- **Cụm địa lý**: country, city, state, nation

**Giải thích tại sao**:
- GloVe học từ thống kê đồng xuất hiện, nên từ xuất hiện cùng ngữ cảnh sẽ có vectors tương tự
- Phép chiếu PCA bảo toàn khoảng cách tương đối, cho phép quan sát các mẫu clustering
- Trực quan hóa 2D tuy mất thông tin nhưng vẫn thể hiện được các mối quan hệ ngữ nghĩa chính

#### 5.5 So sánh Models

| Aspect | Pre-trained GloVe | Custom Word2Vec | Spark MLlib |
|--------|-------------------|-----------------|-------------|
| **Vocabulary** | 400,000 | 3,772 | 78,930 |
| **Training data** | Massive web data | Small EWT corpus | Medium C4 dataset |
| **Quality** | Excellent | Good | Very Good |
| **Similarity scores** | Very high (0.9+) | Moderate (0.8) | High (0.8+) |
| **Semantic coverage** | Comprehensive | Limited | Good |
| **Training time** | N/A | Seconds | Minutes |

**Kết luận**: 
- Pre-trained model có chất lượng tốt nhất do data training lớn
- Custom model với limited data vẫn học được các mối quan hệ ngữ nghĩa cơ bản
- Spark model cân bằng tốt giữa chất lượng và hiệu suất

### 6. Khó khăn và Giải pháp

#### 6.1 Vấn đề bộ nhớ với Dataset lớn
**Khó khăn**: 
- File C4 dataset lớn (30K documents) gây tràn bộ nhớ khi xử lý với sample_fraction=1.0
- Spark tasks thất bại do không đủ bộ nhớ

**Giải pháp**:
- Tối ưu Spark configuration với adaptive execution
- Sử dụng `.cache()` cho việc xử lý DataFrames
- Xử lý lỗi robust để cleanup các tài nguyên hệ thống

#### 6.2 Visualization
**Khó khăn**:
- PCA trên 400K vectors mất thời gian
- Matplotlib hiển thị chậm với các biểu đồ scatter lớn

**Giải pháp**:
- Subset vectors cho visualization thay vì full vocabulary
- Tối ưu plotting parameters (alpha, point size)
- Progressive visualization approach

#### 6.3 Model Compatibility
**Khó khăn**:
- Different vector formats giữa Gensim và Spark MLlib
- Inconsistent APIs for similarity search

**Giải pháp**:
- Wrapper class `WordEmbedder` để unify interface
- Standardized output formatting
- Consistent error handling across platforms

#### 6.4 Output Management
**Khó khăn**:
- Multiple scripts tạo nhiều files với timestamps
- Khó khăn trong việc track latest results

**Giải pháp**:
- Chuyển sang single output file per script
- Ghi đè thay vì append để maintain latest results

### 7. Nguồn tham khảo
- [Stanford GloVe](https://nlp.stanford.edu/projects/glove/)
- [Gensim Documentation](https://radimrehurek.com/gensim/)
- [Scikit-learn PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
- [Apache Spark MLlib Documentation](https://spark.apache.org/docs/latest/ml-guide.html)