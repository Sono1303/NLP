
# Báo cáo Lab5: Text Classification & Sentiment Analysis

Báo cáo và Phân tích (50%)

### 1. Giải thích chi tiết các bước triển khai
- **Tiền xử lý dữ liệu:**
  - Loại bỏ ký tự đặc biệt, URL, HTML tag giúp giảm nhiễu và tăng chất lượng đặc trưng.
  - Chuyển về chữ thường, loại stopwords để chuẩn hóa dữ liệu.
  - Giảm từ vựng theo tần suất (chỉ giữ từ xuất hiện đủ nhiều, loại từ quá hiếm/quá phổ biến).
  - Loại bỏ câu ngắn (dưới 3 từ) để tránh các câu không mang nhiều thông tin.
- **Trích xuất đặc trưng:**
  - TF-IDF: Sử dụng HashingTF và IDF để tạo vector đặc trưng cho văn bản.
  - Word2Vec: Huấn luyện embedding trên tập dữ liệu, lấy trung bình vector từ cho mỗi câu.
- **Huấn luyện mô hình:**
  - LogisticRegression: baseline cho bài toán phân loại nhị phân.
  - NaiveBayes: mô hình xác suất đơn giản, thường hiệu quả với dữ liệu văn bản.
  - GBT: mô hình ensemble mạnh, khai thác tốt các đặc trưng phi tuyến.
  - NeuralNet: mạng nơ-ron nhiều lớp, học được biểu diễn phức tạp của dữ liệu.
- **Đánh giá:**
  - Sử dụng các chỉ số: Accuracy, F1-score, Precision, Recall để so sánh mô hình.

### 2. Hướng dẫn chi tiết cách chạy code
- Đảm bảo đã cài đặt các thư viện: scikit-learn, pyspark.
- Chạy các script kiểm thử bằng lệnh:
  ```
  python test/lab5_test.py
  python test/lab5_spark_sentiment_analysis.py
  python test/lab5_spark_sentiment_analysis_app_1.py
  python test/lab5_spark_sentiment_analysis_app_2.py
  python test/lab5_spark_sentiment_analysis_app_3.py
  python test/lab5_spark_sentiment_analysis_advanced.py
  ```
- Kết quả sẽ được lưu vào thư mục `Lab4/results/` với tên file tương ứng.
- Có thể mở file kết quả để xem các chỉ số đánh giá từng mô hình.

### 3. Phân tích kết quả từng mô hình
- **Kết quả mô hình baseline (LogisticRegression, TF-IDF):**
  - Accuracy: 0.7333
  - F1-score: 0.7316
- **Kết quả mô hình cải tiến:**
  - NaiveBayes (TF-IDF): Accuracy 0.7333, F1 0.7359
  - GBT (TF-IDF): Accuracy 0.7255, F1 0.6910
  - NeuralNet (TF-IDF): Accuracy 0.7637, F1 0.7635
  - LogisticRegression (Word2Vec): Accuracy 0.6529, F1 0.6002
  - GBT (Word2Vec): Accuracy 0.6775, F1 0.6480
  - NeuralNet (Word2Vec): Accuracy 0.6382, F1 0.5115

- **Nhận xét chi tiết:**
  - NeuralNet với TF-IDF cho kết quả tốt nhất, chứng tỏ mạng nơ-ron có khả năng học biểu diễn phức tạp từ đặc trưng TF-IDF.
  - Word2Vec chưa vượt qua TF-IDF trên tập dữ liệu này, có thể do embedding chưa đủ lớn hoặc dữ liệu chưa đa dạng.
  - NaiveBayes và LogisticRegression cho kết quả tương đương, phù hợp với bài toán phân loại văn bản đơn giản.
  - GBT có thể khai thác tốt đặc trưng phi tuyến nhưng cần tối ưu thêm tham số.
  - Tiền xử lý nâng cao giúp tăng chất lượng đặc trưng, loại bỏ nhiễu và dữ liệu không hữu ích.

### 4. Khó khăn thực tế và giải pháp
- **Xử lý label:** Dữ liệu gốc có label -1, 1. Phải chuyển -1 thành 0 để phù hợp với các mô hình Spark ML (yêu cầu label là số nguyên không âm).
- **Phân phối label:** Một số tập dữ liệu có thể bị lệch nhãn, cần kiểm tra phân phối label sau khi lọc.
- **Memory error:** Khi dùng GBT với số chiều đặc trưng lớn, gặp lỗi bộ nhớ. Đã khắc phục bằng cách giảm numFeatures của HashingTF.
- **Chất lượng embedding:** Word2Vec cần dữ liệu lớn và đa dạng để học embedding tốt. Có thể thử pre-trained embedding (GloVe, FastText) nếu muốn cải thiện.
- **Tối ưu pipeline:** Việc kết hợp nhiều bước tiền xử lý, đặc trưng và mô hình cần kiểm tra kỹ để tránh lỗi và đảm bảo dữ liệu đầu vào hợp lệ.

### 5. Tài liệu tham khảo
- [Scikit-learn documentation](https://scikit-learn.org/)
- [PySpark ML documentation](https://spark.apache.org/docs/latest/ml-guide.html)
- [Hướng dẫn xử lý dữ liệu văn bản với Spark ML](https://spark.apache.org/docs/latest/ml-features.html)
- [Word2Vec in Spark ML](https://spark.apache.org/docs/latest/ml-features.html#word2vec)
- [Gradient-Boosted Trees in Spark ML](https://spark.apache.org/docs/latest/ml-classification-regression.html#gradient-boosted-trees-gbts)
- [MultilayerPerceptronClassifier in Spark ML](https://spark.apache.org/docs/latest/ml-classification-regression.html#multilayer-perceptron-classifier)

---

**Kết quả chi tiết các mô hình được lưu tại thư mục `Lab4/results/`. Có thể mở các file kết quả để so sánh từng mô hình, từng loại đặc trưng.**
