# Báo cáo Lab4: Text Classification & Sentiment Analysis

## 1. Source code sử dụng cho báo cáo
- [src/models/text_classifier.py](../Lab4/src/models/text_classifier.py): Classifier Scikit-learn
- [test/lab5_test.py](../Lab4/test/lab5_test.py): Kiểm thử TextClassifier
- [test/lab5_spark_sentiment_analysis.py](../Lab4/test/lab5_spark_sentiment_analysis.py): Pipeline Spark ML cơ bản
- [test/lab5_spark_sentiment_analysis_app_1.py](../Lab4/test/lab5_spark_sentiment_analysis_app_1.py): Pipeline cải tiến preprocessing
- [test/lab5_spark_sentiment_analysis_app_2.py](../Lab4/test/lab5_spark_sentiment_analysis_app_2.py): Pipeline embedding Word2Vec
- [test/lab5_spark_sentiment_analysis_app_3.py](../Lab4/test/lab5_spark_sentiment_analysis_app_3.py): Pipeline mô hình phức tạp
- [test/lab5_spark_sentiment_analysis_advanced.py](../Lab4/test/lab5_spark_sentiment_analysis_advanced.py): Pipeline kết hợp cải tiến
- [Lab4/results/](../Lab4/results/): Kết quả test, training
- [Lab4/data/](../Lab4/data/): Dữ liệu cảm xúc

## 1.1. Dataset sử dụng

**Tên dataset 1**: Twitter Financial News Sentiment

**Mô tả**: Dataset chứa tweets về tin tức tài chính được gán nhãn cảm xúc.

**Cấu trúc dữ liệu**:
- **Format**: CSV
- **Số lượng**: ~11,932 tweets
- **Các cột**:
  - `text`: Nội dung tweet (string)
  - `label`: Nhãn cảm xúc - Negative/Neutral/Positive (string)
- **Phân bố**: Negative (~2,106), Neutral (~5,040), Positive (~4,786)

**Nguồn**: Kaggle - Twitter Financial News Sentiment Dataset

**Tên dataset 2**: Sentiments.csv

**Mô tả**: Dataset cảm xúc đơn giản cho classification.

**Cấu trúc dữ liệu**:
- **Format**: CSV
- **Các cột**:
  - `text`: Văn bản (string)
  - `sentiment`: Nhãn cảm xúc (string/integer)

**Lưu ý**: Datasets không được commit lên GitHub. Xem [data/README.md](../data/README.md) để biết chi tiết.

### 2. Giải thích chi tiết các bước triển khai

**Task 1: Scikit-learn TextClassifier**
- Xây dựng class `TextClassifier` trong `src/models/text_classifier.py` với các phương thức:
  - `fit`: Huấn luyện mô hình LogisticRegression trên dữ liệu văn bản đã vector hóa.
  - `predict`: Sinh dự đoán cho tập dữ liệu mới.
  - `evaluate`: Tính toán các chỉ số đánh giá (Accuracy, Precision, Recall, F1-score).

**Task 2: Basic Test Case**
- Tạo file kiểm thử `test/lab5_test.py`:
  - Chia nhỏ tập dữ liệu thành train/test.
  - Tiền xử lý văn bản bằng RegexTokenizer và CountVectorizer.
  - Huấn luyện, dự đoán và đánh giá mô hình TextClassifier trên dữ liệu mẫu.

**Task 3: Running the Spark Example**
- Chạy script `test/lab5_spark_sentiment_analysis.py`:
  - Đọc dữ liệu cảm xúc từ file CSV.
  - Tiền xử lý: chuẩn hóa nhãn, loại bỏ NA, tách train/test.
  - Xây dựng pipeline Spark ML gồm các bước: Tokenizer, StopWordsRemover, HashingTF, IDF, LogisticRegression.
  - Huấn luyện và đánh giá mô hình trên dữ liệu lớn với Spark.

**Task 4: Model Improvement Experiment**
- Thực hiện các cải tiến mô hình:
  - Tiền xử lý nâng cao: loại nhiễu, giảm từ vựng, loại câu ngắn.
  - Sử dụng đặc trưng Word2Vec hoặc thử nghiệm mô hình khác như NaiveBayes, GBT, NeuralNet.
  - Chạy kiểm thử với script nâng cao (`test/lab5_spark_sentiment_analysis_advanced.py`) để so sánh kết quả các mô hình và đặc trưng.

### 3. Hướng dẫn chi tiết cách chạy code
- Đảm bảo đã cài đặt các thư viện: scikit-learn, pyspark.
- Chạy các script kiểm thử bằng lệnh:
  ```
  # Task 2: Basic Test Case
  python test/lab5_test.py                # Kiểm thử TextClassifier với dữ liệu nhỏ, kiểm tra fit/predict/evaluate

  # Task 3: Running the Spark Example
  python test/lab5_spark_sentiment_analysis.py      # Pipeline Spark ML cơ bản với LogisticRegression
  python test/lab5_spark_sentiment_analysis_app_1.py # Biến thể pipeline, kiểm thử cải tiến preprocessing (lọc nhiễu, giảm từ vựng, giảm chiều đặc trưng)
  python test/lab5_spark_sentiment_analysis_app_2.py # Biến thể pipeline, kiểm thử embedding Word2Vec thay cho TF-IDF
  python test/lab5_spark_sentiment_analysis_app_3.py # Biến thể pipeline, kiểm thử các mô hình phức tạp hơn (NaiveBayes, GBT, NeuralNet)

  # Task 4: Model Improvement Experiment
  python test/lab5_spark_sentiment_analysis_advanced.py # Pipeline kết hợp các cải tiến: preprocessing nâng cao, embedding (Word2Vec), mô hình phức tạp (NaiveBayes, GBT, NeuralNet)
  ```
- Kết quả sẽ được lưu vào thư mục `Lab4/results/` với tên file tương ứng.
- Có thể mở file kết quả để xem các chỉ số đánh giá từng mô hình.


### 4. Phân tích kết quả

**Task 2: Basic Test Case**
- File: `test/lab5_test.py`
- Mô hình LogisticRegression với CountVectorizer trên dữ liệu nhỏ.
- Kết quả kiểm thử:
```
Model training time: 0.0000 seconds
Model prediction time: 0.0000 seconds
Evaluation metrics:
accuracy: 0.500
precision: 0.500
recall: 1.000
f1: 0.667
```
- Nhận xét & Giải thích:
  - Accuracy và precision đều đạt 0.5, recall đạt 1.0, f1 đạt 0.667. Điều này cho thấy mô hình dự đoán đúng toàn bộ các mẫu thuộc một lớp (recall=1.0), nhưng precision thấp do dự đoán nhầm sang lớp còn lại.
  - Nguyên nhân có thể do tập dữ liệu kiểm thử rất nhỏ, phân phối nhãn không cân bằng hoặc số lượng mẫu quá ít khiến mô hình chưa học được ranh giới phân loại rõ ràng.
  - Kết quả này phù hợp với mục tiêu kiểm thử chức năng (fit, predict, evaluate) của class TextClassifier, không nhằm tối ưu hóa độ chính xác mà để xác nhận pipeline hoạt động đúng.
  - Đây là baseline tối thiểu trước khi áp dụng các pipeline lớn hơn với Spark và dữ liệu thực tế.

**Task 3: Baseline Model (LogisticRegression, TF-IDF)**
- File: `test/lab5_spark_sentiment_analysis.py`
- Mô hình baseline sử dụng TF-IDF để vector hóa văn bản và LogisticRegression để phân loại.
- Kết quả:
```
Model training time: 4.6797 seconds
Model evaluation time: 0.9917 seconds
Test Accuracy: 0.7295
Test F1 Score: 0.7266
```
- Nhận xét & Giải thích:
  - Accuracy ~0.73 và F1 ~0.73 cho thấy mô hình baseline đã học được các đặc trưng quan trọng từ văn bản nhờ TF-IDF, phân biệt được các nhãn cảm xúc ở mức khá tốt.
  - Thời gian huấn luyện và đánh giá nhanh nhờ Spark ML pipeline tối ưu hóa trên dữ liệu lớn.
  - Tuy nhiên, kết quả này cũng phản ánh giới hạn của mô hình tuyến tính (LogisticRegression) và đặc trưng TF-IDF: mô hình chỉ tận dụng tần suất từ, chưa khai thác được ngữ nghĩa sâu hoặc mối quan hệ giữa các từ.
  - Đây là chuẩn để so sánh với các phương pháp cải tiến về preprocessing, embedding hoặc mô hình phức tạp hơn ở các task sau.

**Task 4: Improved Models and Techniques**

*A. Cải tiến Preprocessing & Feature Selection*
- File: `test/lab5_spark_sentiment_analysis_app_1.py`
- Áp dụng lọc nhiễu, giảm từ vựng, giảm chiều đặc trưng TF-IDF.
- Mục tiêu: Giảm noise, tăng chất lượng đặc trưng, giúp mô hình tổng quát tốt hơn.
- Kết quả thực tế:
  ```
  Model training time: 17.3923 seconds
  Model evaluation time: 0.0890 seconds
  Test Accuracy: 0.7366
  Test F1 Score: 0.7369
  ```
- Nhận xét & Giải thích:
  - Accuracy và F1 tăng nhẹ so với baseline (0.7366 vs 0.7295), cho thấy preprocessing nâng cao giúp mô hình tổng quát tốt hơn, giảm nhiễu.
  - Thời gian huấn luyện tăng do pipeline phức tạp hơn.
  - Đây là bước cải tiến hiệu quả, nhưng mức tăng còn hạn chế do đặc trưng vẫn dựa trên TF-IDF.

*B. Sử dụng Embedding Word2Vec*
- File: `test/lab5_spark_sentiment_analysis_app_2.py`
- Thay thế TF-IDF bằng Word2Vec để biểu diễn văn bản.
- Kết quả thực tế:
  ```
  Model training time: 5.8546 seconds
  Model evaluation time: 0.0536 seconds
  Test Accuracy: 0.6573
  Test F1 Score: 0.5921
  ```
- Nhận xét & Giải thích:
  - Độ chính xác và F1 giảm rõ rệt so với TF-IDF (F1: 0.5921 so với 0.7369), cho thấy việc chỉ lấy trung bình vector từ làm mất nhiều thông tin ngữ cảnh.
  - Word2Vec có tiềm năng biểu diễn ngữ nghĩa, nhưng cần kỹ thuật tổng hợp tốt hơn hoặc embedding lớn hơn để phát huy hiệu quả.
  - Thời gian train nhanh hơn do số chiều đặc trưng giảm.

*C. Thử nghiệm các mô hình phức tạp hơn*
- File: `test/lab5_spark_sentiment_analysis_app_3.py`
- Sử dụng các mô hình: NaiveBayes, GBT, NeuralNet trên đặc trưng TF-IDF.
Kết quả thực tế:

| Mô hình                   | TrainTime(s) | EvalTime(s) | Accuracy | F1    |
|---------------------------|--------------|-------------|----------|-------|
| Bag-of-Words + NaiveBayes | 1.87         | 0.08        | 0.7115   | 0.7074|
| Bag-of-Words + GBT        | 4.67         | 0.04        | 0.7024   | 0.6630|
| Bag-of-Words + NeuralNet  | 6.95         | 0.04        | 0.7403   | 0.7274|

- Nhận xét & Giải thích:
  - NeuralNet cho F1 cao nhất (0.7274), vượt nhẹ baseline, cho thấy mô hình phi tuyến có thể khai thác tốt hơn đặc trưng TF-IDF.
  - NaiveBayes và GBT cho kết quả tương đương baseline, phù hợp với đặc thù dữ liệu văn bản.
  - Thời gian train tăng dần theo độ phức tạp mô hình.

*D. Kết hợp nhiều cải tiến (Advanced Pipeline)*
- File: `test/lab5_spark_sentiment_analysis_advanced.py`
- Kết hợp preprocessing nâng cao, embedding (Word2Vec), và các mô hình phức tạp.
Kết quả thực tế:

| Mô hình                  | TrainTime(s) | EvalTime(s) | Accuracy | F1    |
|--------------------------|--------------|-------------|----------|-------|
| LogisticRegression       | 8.03         | 0.05        | 0.7333   | 0.7316|
| NaiveBayes               | 3.94         | 0.06        | 0.7333   | 0.7359|
| GBT                      | 12.07        | 0.04        | 0.7255   | 0.6910|
| NeuralNet                | 7.73         | 0.03        | 0.7637   | 0.7635|
| LogisticRegression_W2V   | 8.45         | 0.05        | 0.6529   | 0.6002|
| GBT_W2V                  | 13.50        | 0.04        | 0.6775   | 0.6480|
| NeuralNet_W2V            | 6.93         | 0.05        | 0.6382   | 0.5115|

- Nhận xét & Giải thích:
  - NeuralNet với TF-IDF cho F1 cao nhất (0.7635), vượt rõ rệt các mô hình khác, chứng tỏ mô hình phi tuyến và đặc trưng TF-IDF phù hợp nhất với bài toán này.
  - Word2Vec khi chỉ lấy trung bình vector từ cho kết quả thấp hơn TF-IDF ở mọi mô hình.
  - NaiveBayes và LogisticRegression cho kết quả ổn định, GBT cần tối ưu thêm tham số.
  - Kết hợp nhiều cải tiến giúp kiểm chứng pipeline tối ưu, nhưng TF-IDF + NeuralNet vẫn là lựa chọn tốt nhất trên tập dữ liệu này.

**So sánh và phân tích hiệu quả cải tiến**
**Tổng hợp so sánh các pipeline và mô hình:**
- Preprocessing nâng cao (lọc nhiễu, giảm từ vựng) giúp tăng nhẹ độ chính xác và F1 so với baseline, nhưng hiệu quả còn hạn chế nếu đặc trưng vẫn dựa trên TF-IDF.
- Word2Vec khi chỉ lấy trung bình vector từ cho kết quả thấp hơn TF-IDF ở mọi mô hình, do mất thông tin ngữ cảnh. Để phát huy sức mạnh embedding, cần kỹ thuật tổng hợp tốt hơn (attention, sequence model) hoặc dùng embedding lớn hơn, pre-trained.
- NeuralNet với TF-IDF consistently cho F1 cao nhất (0.7635), vượt rõ rệt các mô hình tuyến tính và các đặc trưng khác. Điều này cho thấy mô hình phi tuyến có khả năng học biểu diễn phức tạp, tận dụng tốt đặc trưng tần suất từ.
- NaiveBayes và LogisticRegression cho kết quả ổn định, phù hợp với dữ liệu văn bản, nhưng bị giới hạn bởi tính tuyến tính và không tận dụng được quan hệ phi tuyến giữa các đặc trưng.
- GBT có tiềm năng nhưng cần tối ưu thêm tham số, hiện tại chưa vượt được NeuralNet.
- Kết hợp nhiều cải tiến giúp kiểm chứng pipeline tối ưu, nhưng trên tập dữ liệu này, TF-IDF + NeuralNet vẫn là lựa chọn tốt nhất về hiệu quả tổng thể.
- Thời gian huấn luyện tăng dần theo độ phức tạp pipeline và mô hình, cần cân nhắc trade-off giữa hiệu quả và chi phí tính toán khi triển khai thực tế.

### 5. Khó khăn thực tế và giải pháp
- **Xử lý label:** Dữ liệu gốc có label -1, 1. Phải chuyển -1 thành 0 để phù hợp với các mô hình Spark ML (yêu cầu label là số nguyên không âm).
- **Phân phối label:** Một số tập dữ liệu có thể bị lệch nhãn, cần kiểm tra phân phối label sau khi lọc.
- **Memory error:** Khi dùng GBT với số chiều đặc trưng lớn, gặp lỗi bộ nhớ. Đã khắc phục bằng cách giảm numFeatures của HashingTF.
- **Chất lượng embedding:** Word2Vec cần dữ liệu lớn và đa dạng để học embedding tốt. Có thể thử pre-trained embedding (GloVe, FastText) nếu muốn cải thiện.
- **Tối ưu pipeline:** Việc kết hợp nhiều bước tiền xử lý, đặc trưng và mô hình cần kiểm tra kỹ để tránh lỗi và đảm bảo dữ liệu đầu vào hợp lệ.
 - **Tuning hyperparameter:** Việc chọn tham số tối ưu cho các mô hình (như learning rate, số chiều embedding, số layer NeuralNet, maxIter...) ảnh hưởng lớn đến kết quả. Cần thử nghiệm grid search hoặc random search để tìm cấu hình tốt nhất.
 - **Reproducibility:** Kết quả có thể thay đổi giữa các lần chạy do random seed, chia dữ liệu train/test khác nhau. Nên cố định seed và ghi rõ quy trình chia dữ liệu để đảm bảo tái lập.
 - **Dữ liệu thiếu hoặc lỗi:** Một số dòng dữ liệu có thể bị thiếu trường hoặc lỗi định dạng, cần kiểm tra và loại bỏ kỹ trước khi huấn luyện.
 - **Triển khai thực tế:** Khi áp dụng trên dữ liệu lớn hơn hoặc môi trường production, cần cân nhắc về tài nguyên tính toán, thời gian huấn luyện, và khả năng mở rộng (scaling) của pipeline.


### 6. Tài liệu tham khảo
- [Scikit-learn Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [PySpark MLlib Documentation](https://spark.apache.org/docs/latest/ml-guide.html)
- [Word2Vec Paper](https://arxiv.org/abs/1301.3781)
- [TF-IDF Explained](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
- [Neural Networks for Text Classification](https://www.deeplearningbook.org/)
- [CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

