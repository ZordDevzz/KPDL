# Tài liệu hướng dẫn: requirements.txt

Tệp `requirements.txt` liệt kê danh sách các thư viện Python (packages) cần thiết để chạy dự án này.

## Chi tiết các Thư viện

```text
scikit-learn
pandas
matplotlib
seaborn
joblib
```

1.  **pandas**:
    *   **Mục đích:** Thư viện mạnh mẽ nhất để thao tác và phân tích dữ liệu dạng bảng (DataFrame).
    *   **Sử dụng trong dự án:** Đọc file CSV (`pd.read_csv`), làm sạch dữ liệu (drop duplicates, dropna), và xử lý dữ liệu trước khi đưa vào mô hình.

2.  **scikit-learn**:
    *   **Mục đích:** Thư viện Học máy (Machine Learning) tiêu chuẩn.
    *   **Sử dụng trong dự án:**
        *   `preprocessing`: `StandardScaler` (chuẩn hóa), `LabelEncoder` (mã hóa).
        *   `model_selection`: `train_test_split` (chia tập train/test).
        *   `linear_model`: `LinearRegression` (thuật toán chính).
        *   `metrics`: Tính toán sai số (`mean_squared_error`, `r2_score`).

3.  **matplotlib**:
    *   **Mục đích:** Thư viện nền tảng để vẽ biểu đồ 2D.
    *   **Sử dụng trong dự án:** Tạo khung hình (`plt.figure`), lưu ảnh (`plt.savefig`), và tùy chỉnh các yếu tố đồ thị.

4.  **seaborn**:
    *   **Mục đích:** Thư viện vẽ biểu đồ thống kê cấp cao, xây dựng dựa trên matplotlib nhưng đẹp và dễ dùng hơn.
    *   **Sử dụng trong dự án:** Vẽ Heatmap (biểu đồ nhiệt), Scatterplot (biểu đồ tán xạ), Boxplot (biểu đồ hộp), Histplot (biểu đồ phân phối).

5.  **joblib**:
    *   **Mục đích:** Công cụ để lưu và tải các đối tượng Python ra đĩa (serialization), đặc biệt hiệu quả với các mảng dữ liệu lớn.
    *   **Sử dụng trong dự án:** Lưu mô hình đã huấn luyện thành file `.pkl` để có thể sử dụng lại sau này mà không cần huấn luyện lại.
