# Báo cáo Lab4: Text Classification & Sentiment Analysis

### 1. Giải thích chi tiết các bước triển khai theo từng task

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

### 2. Hướng dẫn chi tiết cách chạy code
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


### 3. Phân tích kết quả theo từng task

**Task 1: Scikit-learn TextClassifier**
- File: `src/models/text_classifier.py`, kiểm thử: `test/lab5_test.py`
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
- Nhận xét: Mô hình scikit-learn hoạt động tốt trên dữ liệu nhỏ, kiểm tra đúng chức năng fit/predict/evaluate. Đây là bước xác nhận baseline trước khi chuyển sang Spark.

**Task 2: Basic Test Case**
- File: `test/lab5_test.py`
- Chia nhỏ dữ liệu, tiền xử lý bằng RegexTokenizer và CountVectorizer.
- Kết quả kiểm thử:
```
Model training time: 4.6797 seconds
Model evaluation time: 0.9917 seconds
Test Accuracy: 0.7295
Test F1 Score: 0.7266
```
- Nhận xét: Kiểm thử xác nhận pipeline tiền xử lý và phân loại hoạt động đúng, đảm bảo dữ liệu chia train/test độc lập, không rò rỉ thông tin.

**Task 3: Baseline Model (LogisticRegression, TF-IDF)**
- File: `test/lab5_spark_sentiment_analysis.py`
- Mô hình baseline sử dụng TF-IDF để vector hóa văn bản và LogisticRegression để phân loại.
- Kết quả:
  - Accuracy: 0.7333
  - F1-score: 0.7316
- Nhận xét: TF-IDF giúp mô hình LogisticRegression phân biệt tốt các đặc trưng quan trọng trong văn bản. Đây là chuẩn để so sánh với các phương pháp cải tiến.

**Task 4: Improved Models and Techniques**

*A. Cải tiến Preprocessing & Feature Selection*
- File: `test/lab5_spark_sentiment_analysis_app_1.py`
- Áp dụng lọc nhiễu, giảm từ vựng, giảm chiều đặc trưng TF-IDF.
- Mục tiêu: Giảm noise, tăng chất lượng đặc trưng, giúp mô hình tổng quát tốt hơn.

*B. Sử dụng Embedding Word2Vec*
- File: `test/lab5_spark_sentiment_analysis_app_2.py`
- Thay thế TF-IDF bằng Word2Vec để biểu diễn văn bản.
- Kết quả:
  - LogisticRegression (Word2Vec): Accuracy 0.6529, F1 0.6002
  - GBT (Word2Vec): Accuracy 0.6775, F1 0.6480
  - NeuralNet (Word2Vec): Accuracy 0.6382, F1 0.5115
- Nhận xét: Word2Vec giúp biểu diễn ngữ nghĩa tốt hơn, nhưng khi chỉ lấy trung bình vector từ cho cả câu, thông tin ngữ cảnh bị mất, kết quả chưa vượt qua TF-IDF.

*C. Thử nghiệm các mô hình phức tạp hơn*
- File: `test/lab5_spark_sentiment_analysis_app_3.py`
- Sử dụng các mô hình: NaiveBayes, GBT, NeuralNet trên đặc trưng TF-IDF.
- Kết quả:
  - NaiveBayes (TF-IDF): Accuracy 0.7333, F1 0.7359
  - GBT (TF-IDF): Accuracy 0.7255, F1 0.6910
  - NeuralNet (TF-IDF): Accuracy 0.7637, F1 0.7635
- Nhận xét: NeuralNet học được biểu diễn phức tạp từ TF-IDF, cho kết quả tốt nhất. NaiveBayes phù hợp với dữ liệu văn bản, GBT cần tối ưu thêm tham số.

*D. Kết hợp nhiều cải tiến (Advanced Pipeline)*
- File: `test/lab5_spark_sentiment_analysis_advanced.py`
- Kết hợp preprocessing nâng cao, embedding (Word2Vec), và các mô hình phức tạp.
- Kết quả cho thấy:
  - Tiền xử lý nâng cao giúp loại bỏ nhiễu, tăng chất lượng đặc trưng, đặc biệt với TF-IDF.
  - NeuralNet với TF-IDF vẫn cho kết quả tốt nhất, chứng tỏ đặc trưng TF-IDF phù hợp với bài toán phân loại cảm xúc văn bản này.
  - Word2Vec chưa vượt qua TF-IDF, nhưng có thể cải thiện nếu dùng embedding lớn hơn hoặc kết hợp với các đặc trưng khác.
  - GBT và NeuralNet có thể khai thác tốt đặc trưng phi tuyến, nhưng cần tối ưu thêm tham số và cấu trúc mạng.

**So sánh và phân tích hiệu quả cải tiến**
- Các cải tiến về preprocessing giúp giảm nhiễu, tăng chất lượng đặc trưng đầu vào.
- Word2Vec có tiềm năng nhưng cần kỹ thuật kết hợp hoặc embedding lớn hơn để phát huy hiệu quả.
- NeuralNet cho kết quả tốt nhất nhờ khả năng học biểu diễn phức tạp.
- Việc kết hợp nhiều phương pháp giúp kiểm chứng và chọn ra pipeline tối ưu cho từng bài toán cụ thể.

### 4. Khó khăn thực tế và giải pháp
- **Xử lý label:** Dữ liệu gốc có label -1, 1. Phải chuyển -1 thành 0 để phù hợp với các mô hình Spark ML (yêu cầu label là số nguyên không âm).
- **Phân phối label:** Một số tập dữ liệu có thể bị lệch nhãn, cần kiểm tra phân phối label sau khi lọc.
- **Memory error:** Khi dùng GBT với số chiều đặc trưng lớn, gặp lỗi bộ nhớ. Đã khắc phục bằng cách giảm numFeatures của HashingTF.
- **Chất lượng embedding:** Word2Vec cần dữ liệu lớn và đa dạng để học embedding tốt. Có thể thử pre-trained embedding (GloVe, FastText) nếu muốn cải thiện.
- **Tối ưu pipeline:** Việc kết hợp nhiều bước tiền xử lý, đặc trưng và mô hình cần kiểm tra kỹ để tránh lỗi và đảm bảo dữ liệu đầu vào hợp lệ.

### 5. Tài liệu tham khảo
---

