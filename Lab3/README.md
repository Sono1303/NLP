# Lab 3: Word Embeddings - Báo cáo và Phân tích

## Phần 1: Tổng quan dự án

### Mục tiêu
Lab 3 tập trung vào việc triển khai và phân tích các kỹ thuật Word Embeddings trong xử lý ngôn ngữ tự nhiên, bao gồm:
- Sử dụng pre-trained models (GloVe)
- Huấn luyện custom Word2Vec models
- Xử lý dữ liệu lớn với Apache Spark
- Trực quan hóa word embeddings

### Cấu trúc dự án
```
Lab3/
├── Lab3.ipynb                          # Notebook chính với visualization
├── README.md                           # Báo cáo này
├── data/                              
│   ├── glove.6B/                      # Pre-trained GloVe vectors
│   ├── UD_English-EWT/                # Training data cho custom model
│   └── c4-train.00000-of-01024-30K.json # Large dataset cho Spark
├── src/
│   └── representations/
│       └── word_embedder.py           # WordEmbedder class
├── test/
│   ├── lab4_test.py                   # Test pre-trained model
│   ├── lab4_embedding_training_demo.py # Custom Word2Vec training
│   └── lab4_spark_word2vec_demo.py    # Spark MLlib training
└── results/
    ├── lab4_test_output.txt           # Kết quả test pre-trained
    ├── lab4_training_demo_output.txt  # Kết quả custom training
    ├── lab4_spark_word2vec_output.txt # Kết quả Spark training
    └── word2vec_ewt.model             # Custom trained model
```

## Phần 2: Hướng dẫn thực thi

### Yêu cầu hệ thống
- Python 3.8+
- Jupyter Notebook
- Apache Spark (cho phần Spark MLlib)

### Cài đặt dependencies
```bash
pip install gensim numpy matplotlib scikit-learn pandas
pip install pyspark  # Cho phần Spark
```

### Chạy các thành phần

#### 1. Pre-trained Model Test
```bash
cd Lab3
python test/lab4_test.py
```
**Output**: `results/lab4_test_output.txt`

#### 2. Custom Word2Vec Training
```bash
python test/lab4_embedding_training_demo.py
```
**Output**: `results/lab4_training_demo_output.txt`

#### 3. Spark MLlib Training
```bash
python test/lab4_spark_word2vec_demo.py
```
**Output**: `results/lab4_spark_word2vec_output.txt`

#### 4. Visualization (Jupyter Notebook)
```bash
jupyter notebook Lab3.ipynb
```
Chạy tất cả cells để xem trực quan hóa PCA và phân tích word clusters.

## Phần 3: Phân tích kết quả chi tiết

### 3.1 Pre-trained Model (GloVe) Analysis

#### Kết quả chính
- **Model**: GloVe Wiki Gigaword 50D với 400,000 từ
- **Vector quality**: Vectors có giá trị thực tế, không phải zero vectors
- **Similarity scores**: 
  - king-queen: 0.7839 (rất cao, thể hiện mối quan hệ gender)
  - king-man: 0.5309 (vừa phải, thể hiện mối quan hệ hierarchical)

#### Phân tích từ đồng nghĩa cho "computer"
```
1. computers (0.917)  - Plural form, hoàn hảo
2. software (0.881)   - Related concept, logic
3. technology (0.853) - Broader category, reasonable
4. electronic (0.813) - Hardware relationship
5. internet (0.806)   - Usage context
```

**Nhận xét**: Pre-trained model thể hiện khả năng nắm bắt:
- **Morphological relationships**: computer ↔ computers
- **Semantic fields**: computer ↔ software, technology
- **Conceptual associations**: computer ↔ electronic, internet

### 3.2 Custom Word2Vec Training Analysis

#### Kết quả training
- **Dataset**: 13,572 sentences từ UD English-EWT
- **Training time**: 1.58 seconds
- **Vocabulary**: 3,772 words
- **Vector dimensions**: 50D

#### Chất lượng học được
```
Similarities:
- the-man: 0.575
- man-woman: 0.820
```

**Phân tích**: 
- Model tự train đạt similarity scores hợp lý
- Mối quan hệ gender (man-woman: 0.820) được học tốt
- Limited vocabulary do dataset nhỏ, nhưng quality acceptable

### 3.3 Spark MLlib Large Dataset Analysis

#### Kết quả impressive
- **Dataset**: 29,971 documents từ C4
- **Training time**: 5.85 minutes
- **Vocabulary**: 78,930 words (rất lớn!)
- **Vector dimensions**: 100D

#### Quality comparison
```
Similar to 'computer':
1. computers (0.798)
2. desktop (0.702)
3. laptop (0.680)
4. software (0.672)
```

**Phân tích**:
- Vocabulary 20x lớn hơn custom model
- Semantic relationships rõ ràng: computer → desktop, laptop
- Large dataset training cho kết quả robust hơn

### 3.4 Visualization Analysis (PCA + Scatter Plot)

#### Phương pháp
- **Dimensionality reduction**: PCA từ 100D → 2D
- **Variance explained**: ~10-15% (trade-off acceptable cho visualization)
- **Visualization**: Scatter plot với vector arrows từ origin

#### Phân tích biểu đồ trực quan hóa

##### Clustering observations cho từ "king"
```
Similar words clustered: queen, prince, kingdom, royal...
```

**Kết quả quan sát**:
1. **Spatial clustering**: Các từ semantically related có xu hướng gần nhau trong 2D space
2. **Gender relationships**: "king" và "queen" có khoảng cách hợp lý, thể hiện similar concepts nhưng different gender
3. **Hierarchical relationships**: "prince", "duke" cluster gần "king", thể hiện royal hierarchy
4. **Contextual relationships**: "kingdom", "castle" gần nhau, thể hiện domain context

##### Cụm từ thú vị phát hiện được
- **Royal cluster**: king, queen, prince, royal, kingdom
- **Technology cluster**: computer, software, technology, electronic
- **Geographic cluster**: country, city, state, nation

**Giải thích tại sao**:
- GloVe học từ co-occurrence statistics, nên từ xuất hiện cùng context sẽ có vectors tương tự
- PCA projection bảo toàn relative distances, cho phép quan sát clustering patterns
- 2D visualization tuy mất thông tin nhưng vẫn thể hiện được main semantic relationships

### 3.5 So sánh Models

| Aspect | Pre-trained GloVe | Custom Word2Vec | Spark MLlib |
|--------|-------------------|-----------------|-------------|
| **Vocabulary** | 400,000 | 3,772 | 78,930 |
| **Training data** | Massive web data | Small EWT corpus | Medium C4 dataset |
| **Quality** | Excellent | Good | Very Good |
| **Similarity scores** | Very high (0.9+) | Moderate (0.8) | High (0.8+) |
| **Semantic coverage** | Comprehensive | Limited | Good |
| **Training time** | N/A | Seconds | Minutes |

**Kết luận**: 
- Pre-trained model có quality tốt nhất do massive training data
- Custom model với limited data vẫn học được basic relationships
- Spark model cân bằng tốt giữa quality và training efficiency

## Phần 4: Khó khăn và Giải pháp

### 4.1 Vấn đề Memory với Large Dataset

**Khó khăn**: 
- File C4 dataset lớn (30K documents) gây memory overflow khi xử lý với sample_fraction=1.0
- Spark tasks failed do insufficient memory

**Giải pháp**:
- Tối ưu Spark configuration với adaptive execution
- Sử dụng `.cache()` cho processed DataFrames
- Error handling robust để cleanup resources

### 4.2 Visualization Performance

**Khó khăn**:
- PCA trên 400K vectors mất thời gian
- Matplotlib rendering slow với large scatter plots

**Giải pháp**:
- Subset vectors cho visualization thay vì full vocabulary
- Tối ưu plotting parameters (alpha, point size)
- Progressive visualization approach

### 4.3 Model Compatibility

**Khó khăn**:
- Different vector formats giữa Gensim và Spark MLlib
- Inconsistent APIs for similarity search

**Giải pháp**:
- Wrapper class `WordEmbedder` để unify interface
- Standardized output formatting
- Consistent error handling across platforms

### 4.4 Output Management

**Khó khăn**:
- Multiple scripts tạo nhiều files với timestamps
- Difficult để track latest results

**Giải pháp**:
- Chuyển sang single output file per script
- Overwrite thay vì append để maintain latest results
- Clear naming convention

## Phần 5: Kết luận và Đánh giá

### Thành tựu đạt được
✅ **Hoàn thành 100% requirements (5/5 tasks)**:
1. Pre-trained model usage với excellent results
2. Document embedding implementation
3. Custom Word2Vec training thành công
4. Large-scale Spark MLlib training
5. Professional visualization với PCA + scatter plots

### Insights quan trọng
1. **Scale matters**: Larger datasets → better embeddings quality
2. **Visualization value**: 2D projections reveal semantic structures
3. **Tool diversity**: Different tools (Gensim/Spark) có trade-offs khác nhau
4. **Engineering quality**: Proper error handling và output management quan trọng

### Future improvements
- Implement t-SNE comparison với PCA
- Add quantitative evaluation metrics
- Experiment với different vector dimensions
- Interactive visualization với Plotly

## Phần 6: Tài liệu tham khảo

### Chính thức
1. **Gensim Documentation**: https://radimrehurek.com/gensim/
2. **Apache Spark MLlib Guide**: https://spark.apache.org/docs/latest/ml-guide.html
3. **GloVe: Global Vectors for Word Representation**: Pennington et al., 2014
4. **Word2Vec**: Mikolov et al., 2013

### Kỹ thuật
1. **Scikit-learn PCA**: https://scikit-learn.org/stable/modules/decomposition.html
2. **Matplotlib Visualization**: https://matplotlib.org/stable/tutorials/
3. **NumPy Documentation**: https://numpy.org/doc/stable/

### Dataset sources
1. **GloVe Pre-trained Vectors**: https://nlp.stanford.edu/projects/glove/
2. **Universal Dependencies**: https://universaldependencies.org/
3. **C4 Dataset**: Common Crawl clean text

---

**Tác giả**: Lab3 Implementation Report  
**Ngày**: October 16, 2025  
**Version**: 1.0