# Tài liệu hướng dẫn: handlers/model_eval.py

Tệp `model_eval.py` chứa hàm dùng để đánh giá hiệu suất của mô hình học máy. Việc đánh giá mô hình giúp chúng ta hiểu được mức độ chính xác của mô hình khi đưa ra dự đoán trên dữ liệu mới.

## Chi tiết Mã nguồn

### Hàm `evaluate_model(model, X, y, dataset_name="Test")`

```python
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import config

def evaluate_model(model, X, y, dataset_name="Test"):
    """
    Đánh giá hiệu quả của mô hình trên tập dữ liệu cho trước (X, y).
    Trả về một từ điển chứa các chỉ số đánh giá.
    Lưu các chỉ số và biểu đồ đường hồi quy vào thư mục EVAL_DIR.
    """
    y_pred = model.predict(X)
    
    # Tính toán các chỉ số đánh giá
    mse = mean_squared_error(y, y_pred)
    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    print(f"--- Đánh giá Mô hình (Tập {dataset_name}) ---")
    print(f"R-squared (Độ chính xác): {r2:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    
    # Lưu các chỉ số vào tệp văn bản
    metrics_path = os.path.join(config.EVAL_DIR, f"{dataset_name.lower()}_metrics.txt")
    with open(metrics_path, "w", encoding='utf-8') as f:
        f.write(f"--- Đánh giá Mô hình (Tập {dataset_name}) ---
")
        f.write(f"R-squared (Độ chính xác): {r2:.4f}
")
        f.write(f"Mean Squared Error (MSE): {mse:.4f}
")
        f.write(f"Mean Absolute Error (MAE): {mae:.4f}
")
    print(f"Các chỉ số đánh giá đã được lưu tại {metrics_path}")

    # Đảm bảo y là mảng 1 chiều để vẽ biểu đồ
    y_flat = np.ravel(y)

    # Vẽ biểu đồ Thực tế so với Dự đoán (Trực quan hóa Hồi quy Tuyến tính)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_flat, y=y_pred, alpha=0.6)
    
    # Vẽ đường lý tưởng y=x
    min_val = min(y_flat.min(), y_pred.min())
    max_val = max(y_flat.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Đường lý tưởng (y=x)')
    
    plt.xlabel(f"Thực tế {config.TARGET_COLUMN}")
    plt.ylabel(f"Dự đoán {config.TARGET_COLUMN}")
    plt.title(f"Hồi quy Tuyến tính: Thực tế so với Dự đoán (Tập {dataset_name})")
    plt.legend()
    
    plot_path = os.path.join(config.EVAL_DIR, f"{dataset_name.lower()}_prediction_plot.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Biểu đồ dự đoán đã được lưu tại {plot_path}")

    return {
        "r2": r2,
        "mse": mse,
        "mae": mae
    }
```

*   **Mục đích:** Đánh giá hiệu suất của mô hình Hồi quy Tuyến tính và trực quan hóa kết quả dự đoán.
*   **Tham số:**
    *   `model`: Mô hình đã được huấn luyện.
    *   `X`: Các đặc trưng (features) của tập dữ liệu dùng để đánh giá.
    *   `y`: Giá trị thực tế (nhãn) của tập dữ liệu dùng để đánh giá.
    *   `dataset_name`: Tên của tập dữ liệu (mặc định là "Test"), dùng cho việc in thông báo và đặt tên file.
*   **Quy trình:**
    1.  **Dự đoán:** Sử dụng `model.predict(X)` để tạo ra các giá trị dự đoán (`y_pred`) từ mô hình.
    2.  **Tính toán chỉ số đánh giá:**
        *   `mean_squared_error (MSE)`: Đo lường bình phương trung bình của sai số giữa giá trị thực tế và giá trị dự đoán. Giá trị càng nhỏ càng tốt.
        *   `mean_absolute_error (MAE)`: Đo lường giá trị tuyệt đối trung bình của sai số. Giá trị càng nhỏ càng tốt.
        *   `r2_score (R-squared)`: Đo lường mức độ phù hợp của mô hình với dữ liệu. Giá trị nằm trong khoảng từ 0 đến 1 (hoặc nhỏ hơn 0 nếu mô hình rất tệ), càng gần 1 càng tốt.
    3.  **In và Lưu chỉ số:**
        *   In các chỉ số đánh giá ra console.
        *   Lưu các chỉ số này vào một tệp văn bản (`.txt`) trong thư mục `eval/` (`eval/test_metrics.txt`).
    4.  **Trực quan hóa Thực tế so với Dự đoán:**
        *   Chuyển `y` thành mảng 1 chiều (`y_flat = np.ravel(y)`) để đảm bảo tương thích với thư viện vẽ biểu đồ.
        *   Tạo biểu đồ phân tán (`sns.scatterplot`) với giá trị thực tế trên trục X và giá trị dự đoán trên trục Y.
        *   Vẽ một đường thẳng màu đỏ (`y=x`) đại diện cho "đường lý tưởng", nơi giá trị dự đoán bằng giá trị thực tế. Mô hình càng tốt thì các điểm dữ liệu càng nằm gần đường này.
        *   Đặt tiêu đề, nhãn trục và chú thích cho biểu đồ.
        *   Lưu biểu đồ dưới dạng file ảnh (`.png`) trong thư mục `eval/` (`eval/test_prediction_plot.png`).
*   **Giá trị trả về:** Một từ điển chứa các chỉ số `r2`, `mse`, `mae`.

```