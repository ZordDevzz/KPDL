# Dự án Hồi quy Tuyến tính - Dự đoán Kết quả Học tập

Dự án này xây dựng một quy trình (pipeline) học máy sử dụng thuật toán **Hồi quy Tuyến tính (Linear Regression)** để dự đoán chỉ số hiệu quả học tập (Performance Index) của học sinh dựa trên các yếu tố như giờ học, điểm số trước đó, thời gian ngủ, v.v.

Mã nguồn được thiết kế theo dạng mô-đun, dễ mở rộng và đi kèm với các công cụ phân tích dữ liệu (EDA) và đánh giá mô hình tự động.

## Cấu trúc Dự án

```text
Linear Regression/
├── config.py             # Tệp cấu hình đường dẫn và tham số
├── main.py               # Điểm khởi chạy chính (Entry point)
├── requirements.txt      # Danh sách các thư viện cần thiết
├── data/                 # Chứa dữ liệu
│   ├── raw/              # Dữ liệu thô (Student_Performance.csv)
│   └── processed/        # Dữ liệu đã làm sạch và chuyển đổi
├── eda/                  # Kết quả phân tích dữ liệu (EDA)
│   ├── figures/          # Biểu đồ phân tích
│   └── summary_statistics.csv
├── eval/                 # Kết quả đánh giá mô hình (Metrics & Plots)
├── handlers/             # Các mô-đun xử lý chức năng
│   ├── data_utils.py     # Làm sạch và chuyển đổi dữ liệu
│   ├── eda_extract.py    # Phân tích khám phá dữ liệu
│   └── model_eval.py     # Đánh giá hiệu suất mô hình
├── models/               # Chứa mô hình đã huấn luyện (.pkl)
└── train/                # Logic huấn luyện mô hình
    └── train.py
```

## Cài đặt

Yêu cầu Python 3.8 trở lên. Cài đặt các thư viện phụ thuộc bằng lệnh:

```bash
pip install -r requirements.txt
```

## Hướng dẫn Sử dụng

Chương trình được điều khiển thông qua tệp `main.py` với các tùy chọn dòng lệnh (CLI):

### 1. Chạy Pipeline Mặc định
Chạy toàn bộ quy trình: Làm sạch -> Chuyển đổi -> Huấn luyện -> Đánh giá.
Nếu các bước trước đó đã có kết quả, chương trình sẽ tự động bỏ qua để tiết kiệm thời gian.

```bash
python main.py
```

### 2. Phân tích Khám phá Dữ liệu (EDA)
Chỉ thực hiện phân tích dữ liệu, vẽ biểu đồ và xuất báo cáo vào thư mục `eda/`.

```bash
python main.py -eda
```

### 3. Đánh giá Mô hình (Evaluation)
Chỉ thực hiện đánh giá mô hình dựa trên dữ liệu test đã lưu. Nếu thiếu file, chương trình sẽ tự động chạy lại các bước cần thiết. Kết quả lưu tại `eval/`.

```bash
python main.py eval
```

### 4. Chạy lại từ đầu (Override)
Bắt buộc chương trình xóa các kết quả cũ và chạy lại toàn bộ quy trình từ đầu.

```bash
python main.py -override
```

## Kết quả Đầu ra

*   **Mô hình:** `models/linear_regression_model.pkl`
*   **Biểu đồ phân tích:** `eda/figures/` (Heatmap, Scatter plots, Boxplots)
*   **Đánh giá:** `eval/` (File text chứa chỉ số MSE, R2 và ảnh biểu đồ so sánh Thực tế vs Dự đoán).

---
*Tài liệu hướng dẫn chi tiết code (Docs) sẽ được tạo ở các bước tiếp theo.*
