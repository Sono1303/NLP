
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
  - Giải thích: TF-IDF là phương pháp vector hóa truyền thống, giúp mô hình LogisticRegression phân biệt tốt các đặc trưng quan trọng trong văn bản. Kết quả này là chuẩn để so sánh với các phương pháp khác.

- **Kết quả mô hình cải tiến:**
  - NaiveBayes (TF-IDF): Accuracy 0.7333, F1 0.7359
    - Giải thích: NaiveBayes phù hợp với dữ liệu văn bản, đặc biệt khi đặc trưng là tần suất từ. Kết quả tương đương LogisticRegression, cho thấy TF-IDF vẫn là đặc trưng mạnh với bài toán này.
  - GBT (TF-IDF): Accuracy 0.7255, F1 0.6910
    - Giải thích: GBT là mô hình ensemble mạnh, có thể khai thác tốt các đặc trưng phi tuyến. Tuy nhiên, với đặc trưng TF-IDF, kết quả chưa vượt trội, có thể do cần tối ưu thêm tham số hoặc dữ liệu chưa đủ lớn.
  - NeuralNet (TF-IDF): Accuracy 0.7637, F1 0.7635
    - Giải thích: NeuralNet học được biểu diễn phức tạp từ TF-IDF, cho kết quả tốt nhất. Điều này chứng tỏ mạng nơ-ron có khả năng khai thác sâu các đặc trưng văn bản khi dữ liệu đủ lớn và tiền xử lý tốt.

- **Kết quả với Word2Vec:**
  - LogisticRegression (Word2Vec): Accuracy 0.6529, F1 0.6002
  - GBT (Word2Vec): Accuracy 0.6775, F1 0.6480
  - NeuralNet (Word2Vec): Accuracy 0.6382, F1 0.5115
  - Giải thích: Word2Vec là phương pháp embedding hiện đại, giúp biểu diễn từ theo ngữ nghĩa. Tuy nhiên, khi chỉ lấy trung bình vector từ cho cả câu, thông tin ngữ cảnh bị mất, dẫn đến kết quả chưa vượt qua TF-IDF. Có thể thử các phương pháp kết hợp hoặc dùng pre-trained embedding để cải thiện.

- **Phương pháp kết hợp (Advanced Preprocessing + TF-IDF + Word2Vec + nhiều mô hình):**
  - Khi kết hợp tiền xử lý nâng cao, giảm từ vựng, loại câu ngắn, và thử nghiệm nhiều mô hình trên cả TF-IDF và Word2Vec, kết quả cho thấy:
    - Tiền xử lý nâng cao giúp loại bỏ nhiễu, tăng chất lượng đặc trưng, đặc biệt với TF-IDF.
    - NeuralNet với TF-IDF vẫn cho kết quả tốt nhất, chứng tỏ đặc trưng TF-IDF phù hợp với bài toán phân loại cảm xúc văn bản này.
    - Word2Vec chưa vượt qua TF-IDF, nhưng có thể cải thiện nếu dùng embedding lớn hơn hoặc kết hợp với các đặc trưng khác.
    - GBT và NeuralNet có thể khai thác tốt đặc trưng phi tuyến, nhưng cần tối ưu thêm tham số và cấu trúc mạng.

- **Nhận xét tổng quan:**
  - TF-IDF vẫn là đặc trưng mạnh cho bài toán phân loại văn bản, đặc biệt khi kết hợp với tiền xử lý nâng cao.
  - NeuralNet cho kết quả tốt nhất nhờ khả năng học biểu diễn phức tạp.
  - Word2Vec phù hợp với các bài toán cần hiểu ngữ nghĩa sâu, nhưng cần kỹ thuật kết hợp hoặc embedding lớn hơn để phát huy hiệu quả.
  - Việc kết hợp nhiều phương pháp giúp kiểm chứng và chọn ra pipeline tối ưu cho từng bài toán cụ thể.

### 4. Khó khăn thực tế và giải pháp
- **Xử lý label:** Dữ liệu gốc có label -1, 1. Phải chuyển -1 thành 0 để phù hợp với các mô hình Spark ML (yêu cầu label là số nguyên không âm).
- **Phân phối label:** Một số tập dữ liệu có thể bị lệch nhãn, cần kiểm tra phân phối label sau khi lọc.
- **Memory error:** Khi dùng GBT với số chiều đặc trưng lớn, gặp lỗi bộ nhớ. Đã khắc phục bằng cách giảm numFeatures của HashingTF.
- **Chất lượng embedding:** Word2Vec cần dữ liệu lớn và đa dạng để học embedding tốt. Có thể thử pre-trained embedding (GloVe, FastText) nếu muốn cải thiện.
- **Tối ưu pipeline:** Việc kết hợp nhiều bước tiền xử lý, đặc trưng và mô hình cần kiểm tra kỹ để tránh lỗi và đảm bảo dữ liệu đầu vào hợp lệ.

### 5. Tài liệu tham khảo
---

