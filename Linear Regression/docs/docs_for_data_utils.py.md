# Tài liệu hướng dẫn: handlers/data_utils.py

Tệp `data_utils.py` chứa các hàm tiện ích để xử lý và chuẩn bị dữ liệu. Đây là bước quan trọng trong mọi dự án học máy để đảm bảo dữ liệu sạch và ở định dạng phù hợp cho mô hình.

## Chi tiết Mã nguồn

### 1. Hàm `data_cleaning(df)`
```python
import pandas as pd
# ... (imports khác)

def data_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Loại bỏ các bản ghi bị thiếu hoặc bị lỗi khỏi DataFrame.
    """
    # Loại bỏ các hàng trùng lặp
    df = df.drop_duplicates()
    
    # Loại bỏ các hàng có giá trị bị thiếu (NaN)
    df = df.dropna()
    
    return df
```
*   **Mục đích:** Đảm bảo chất lượng dữ liệu bằng cách xử lý các vấn đề phổ biến:
    *   `df.drop_duplicates()`: Loại bỏ các hàng có tất cả các giá trị giống hệt nhau, tránh làm sai lệch phân tích hoặc huấn luyện mô hình.
    *   `df.dropna()`: Loại bỏ các hàng chứa ít nhất một giá trị `NaN` (Not a Number - giá trị thiếu). Điều này rất quan trọng vì hầu hết các mô hình học máy không thể xử lý giá trị thiếu.
*   **Đầu vào:** Một DataFrame `df`.
*   **Đầu ra:** Một DataFrame đã được làm sạch.

### 2. Hàm `data_transform(df)`
```python
# ... (imports)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import config

def data_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Chuẩn hóa các dữ liệu cần thiết và mã hóa các biến phân loại để huấn luyện mô hình Hồi quy Tuyến tính.
    """
    df = df.copy() # Tạo bản sao để không ảnh hưởng đến DataFrame gốc
    
    # Mã hóa các cột phân loại (Categorical columns)
    le = LabelEncoder()
    for col in config.CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = le.fit_transform(df[col])
            
    # Chuẩn hóa các cột số (Numerical columns) (trừ cột mục tiêu)
    scaler = StandardScaler()
    cols_to_scale = [col for col in config.NUMERICAL_COLUMNS if col in df.columns and col != config.TARGET_COLUMN]
    
    if cols_to_scale:
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        
    return df
```
*   **Mục đích:** Chuyển đổi dữ liệu sang định dạng mà mô hình Hồi quy Tuyến tính có thể xử lý hiệu quả.
*   **Label Encoding (Mã hóa nhãn):**
    *   `LabelEncoder()`: Chuyển đổi các giá trị văn bản trong cột phân loại thành các số nguyên (ví dụ: "Yes" -> 1, "No" -> 0). Mô hình học máy chỉ làm việc với dữ liệu số.
    *   Duyệt qua `config.CATEGORICAL_COLUMNS` để áp dụng.
*   **Standard Scaling (Chuẩn hóa):**
    *   `StandardScaler()`: Biến đổi dữ liệu số sao cho chúng có giá trị trung bình bằng 0 và độ lệch chuẩn bằng 1.
    *   **Tại sao cần?** Hồi quy Tuyến tính (và nhiều thuật toán khác) rất nhạy cảm với thang đo của dữ liệu. Chuẩn hóa giúp tất cả các đặc trưng đóng góp tương đương vào mô hình, ngăn chặn các đặc trưng có giá trị lớn hơn chi phối quá trình huấn luyện.
    *   Duyệt qua `config.NUMERICAL_COLUMNS` và loại trừ cột mục tiêu (`TARGET_COLUMN`).
*   **Đầu vào:** Một DataFrame `df` đã làm sạch.
*   **Đầu ra:** Một DataFrame đã được chuyển đổi, sẵn sàng cho việc huấn luyện mô hình.
