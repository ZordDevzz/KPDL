# Tài liệu hướng dẫn: config.py

Tệp `config.py` đóng vai trò là "trung tâm điều khiển" của toàn bộ dự án. Nó chứa các định nghĩa đường dẫn thư mục và các cấu hình hằng số cho dữ liệu. Việc tập trung cấu hình vào một nơi giúp mã nguồn sạch sẽ và dễ dàng thay đổi (ví dụ: đổi tên file dữ liệu) mà không cần sửa code ở nhiều nơi.

## Chi tiết Mã nguồn

### 1. Thiết lập Đường dẫn Gốc
```python
import os

# Define project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
```
*   `os.path.abspath(__file__)`: Lấy đường dẫn tuyệt đối của file `config.py` hiện tại.
*   `os.path.dirname(...)`: Lấy thư mục chứa file này. Đây chính là thư mục gốc của dự án.
*   **Mục đích:** Giúp code chạy đúng bất kể bạn đặt thư mục dự án ở đâu trên máy tính.

### 2. Định nghĩa Cấu trúc Thư mục
```python
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
# ... các thư mục khác (EDA, FIGURES, MODEL, EVAL, LOG)
```
*   `os.path.join`: Hàm nối các thành phần đường dẫn một cách an toàn (tự động xử lý dấu `\` trên Windows hoặc `/` trên Linux).
*   Các biến này định nghĩa nơi lưu trữ dữ liệu thô, dữ liệu đã xử lý, biểu đồ, mô hình và kết quả đánh giá.

### 3. Tự động Tạo Thư mục
```python
# Ensure all directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, ...]:
    os.makedirs(directory, exist_ok=True)
```
*   Vòng lặp này duyệt qua danh sách các thư mục quan trọng.
*   `os.makedirs(..., exist_ok=True)`: Tạo thư mục nếu nó chưa tồn tại. Nếu đã tồn tại, không làm gì cả (không báo lỗi).
*   **Lợi ích:** Bạn không cần phải tạo thủ công các thư mục này khi chạy dự án lần đầu.

### 4. Cấu hình Dữ liệu
```python
# Configure column
TARGET_COLUMN = "Performance Index"
NUMERICAL_COLUMNS = [
    "Hours Studied",
    "Previous Scores",
    "Sleep Hours",
    "Sample Question Papers Practiced",
]
CATEGORICAL_COLUMNS = ["Extracurricular Activities"]
```
*   `TARGET_COLUMN`: Tên cột mà chúng ta muốn dự đoán (nhãn).
*   `NUMERICAL_COLUMNS`: Danh sách các cột chứa dữ liệu số (cần chuẩn hóa).
*   `CATEGORICAL_COLUMNS`: Danh sách các cột chứa dữ liệu phân loại (cần mã hóa thành số).
*   **Tại sao cần khai báo ở đây?** Các file khác như `data_utils.py`, `eda_extract.py` sẽ import các biến này để biết cột nào cần xử lý, giúp tránh việc gõ lại tên cột nhiều lần (dễ gây lỗi chính tả).

```