# Tài liệu hướng dẫn: train/train.py

Tệp `train.py` chịu trách nhiệm xây dựng, huấn luyện mô hình và lưu trữ các thành phần cần thiết cho việc đánh giá và sử dụng sau này.

## Chi tiết Mã nguồn

### Hàm `train_model()`

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
import os
import config

def train_model():
    """
    Xây dựng và huấn luyện mô hình Hồi quy Tuyến tính.
    Lưu mô hình và tập dữ liệu kiểm tra (test).
    """
    if not os.path.exists(config.TRANFORMED_DATA_PATH):
        raise FileNotFoundError(f"Dữ liệu đã chuyển đổi không tìm thấy tại {config.TRANFORMED_DATA_PATH}. Hãy chạy pipeline trước.")

    print("Đang tải dữ liệu để huấn luyện...")
    df = pd.read_csv(config.TRANFORMED_DATA_PATH)
    
    if config.TARGET_COLUMN not in df.columns:
        raise ValueError(f"Cột mục tiêu '{config.TARGET_COLUMN}' không tìm thấy trong dữ liệu.")

    X = df.drop(columns=[config.TARGET_COLUMN])
    y = df[config.TARGET_COLUMN]
    
    # Chia dữ liệu
    # Sử dụng random_state để đảm bảo tính tái lập
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Đang huấn luyện mô hình Hồi quy Tuyến tính...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Lưu mô hình
    model_path = os.path.join(config.MODEL_DIR, "linear_regression_model.pkl")
    joblib.dump(model, model_path)
    print(f"Mô hình đã được lưu tại {model_path}")
    
    # Lưu dữ liệu kiểm tra để đánh giá độc lập
    X_test_path = os.path.join(config.PROCESSED_DATA_DIR, "X_test.csv")
    y_test_path = os.path.join(config.PROCESSED_DATA_DIR, "y_test.csv")
    X_test.to_csv(X_test_path, index=False)
    y_test.to_csv(y_test_path, index=False)
    print(f"Dữ liệu kiểm tra đã được lưu tại {config.PROCESSED_DATA_DIR}")
    
    return model, X_test, y_test
```

*   **Mục đích:** Xây dựng và huấn luyện mô hình Hồi quy Tuyến tính, sau đó lưu mô hình và các tập dữ liệu kiểm tra đã chia để sử dụng cho việc đánh giá độc lập sau này.
*   **Quy trình:**
    1.  **Kiểm tra và Tải dữ liệu:**
        *   Kiểm tra xem file dữ liệu đã chuyển đổi (`config.TRANFORMED_DATA_PATH`) có tồn tại không. Nếu không, báo lỗi.
        *   Tải dữ liệu đã chuyển đổi vào DataFrame `df`.
        *   Kiểm tra sự tồn tại của cột mục tiêu (`config.TARGET_COLUMN`).
    2.  **Tách Đặc trưng (X) và Nhãn (y):**
        *   `X` sẽ chứa tất cả các cột ngoại trừ cột mục tiêu.
        *   `y` sẽ là cột mục tiêu.
    3.  **Chia Dữ liệu Train/Test:**
        *   `train_test_split(X, y, test_size=0.2, random_state=42)`: Chia dữ liệu `X` và `y` thành hai phần: 80% cho huấn luyện (`X_train`, `y_train`) và 20% cho kiểm tra (`X_test`, `y_test`).
        *   `test_size=0.2`: 20% dữ liệu dùng làm tập kiểm tra.
        *   `random_state=42`: Đảm bảo rằng việc chia dữ liệu là như nhau mỗi lần chạy, giúp kết quả có tính tái lập.
    4.  **Huấn luyện Mô hình:**
        *   Tạo một đối tượng `LinearRegression()`.
        *   `model.fit(X_train, y_train)`: Huấn luyện mô hình sử dụng tập dữ liệu huấn luyện. Mô hình sẽ tìm ra mối quan hệ tuyến tính giữa `X_train` và `y_train`.
    5.  **Lưu Mô hình:**
        *   `joblib.dump(model, model_path)`: Lưu mô hình đã huấn luyện vào file `linear_regression_model.pkl` trong thư mục `models/`.
        *   **Ý nghĩa:** Có thể tải lại mô hình này sau để sử dụng mà không cần huấn luyện lại.
    6.  **Lưu Dữ liệu Kiểm tra:**
        *   Lưu `X_test` và `y_test` vào các file CSV riêng biệt (`X_test.csv`, `y_test.csv`) trong thư mục `data/processed/`.
        *   **Ý nghĩa:** Đảm bảo rằng cùng một tập dữ liệu kiểm tra luôn được sử dụng để đánh giá mô hình, ngay cả khi chương trình được chạy lại ở chế độ đánh giá độc lập.
*   **Giá trị trả về:** Mô hình đã huấn luyện (`model`), tập dữ liệu đặc trưng kiểm tra (`X_test`), và tập dữ liệu nhãn kiểm tra (`y_test`).
