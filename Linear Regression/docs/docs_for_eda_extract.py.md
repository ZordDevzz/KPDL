# Tài liệu hướng dẫn: handlers/eda_extract.py

Tệp `eda_extract.py` chứa hàm thực hiện **Phân tích Khám phá Dữ liệu (Exploratory Data Analysis - EDA)**. EDA là một bước quan trọng để hiểu cấu trúc, phân phối và mối quan hệ giữa các biến trong dữ liệu, giúp đưa ra các quyết định sáng suốt trong quá trình xây dựng mô hình.

## Chi tiết Mã nguồn

### Hàm `perform_eda(df)`

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import config

def perform_eda(df: pd.DataFrame):
    """
    Thực hiện Phân tích Khám phá Dữ liệu (EDA) và xuất kết quả vào thư mục EDA.
    """
    print("Bắt đầu phân tích EDA...")
    
    # 1. Thống kê mô tả
    summary = df.describe()
    summary_path = os.path.join(config.EDA_DIR, "summary_statistics.csv")
    summary.to_csv(summary_path)
    print(f"Thống kê mô tả đã được lưu tại {summary_path}")
    
    # 2. Biểu đồ nhiệt tương quan (Correlation Heatmap)
    plt.figure(figsize=(12, 10))
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Biểu đồ nhiệt Tương quan')
    plt.tight_layout() # Sửa lỗi tràn văn bản
    heatmap_path = os.path.join(config.FIGURES_DIR, "correlation_heatmap.png")
    plt.savefig(heatmap_path)
    plt.close()
    print(f"Biểu đồ nhiệt tương quan đã được lưu tại {heatmap_path}")
    
    # 3. Phân phối của biến mục tiêu
    plt.figure(figsize=(8, 6))
    sns.histplot(df[config.TARGET_COLUMN], kde=True)
    plt.title(f'Phân phối của {config.TARGET_COLUMN}')
    plt.tight_layout()
    dist_path = os.path.join(config.FIGURES_DIR, "target_distribution.png")
    plt.savefig(dist_path)
    plt.close()
    print(f"Biểu đồ phân phối biến mục tiêu đã được lưu tại {dist_path}")
    
    # 4. Phân tích đặc trưng (Vẽ biểu đồ riêng cho từng thuộc tính)
    print("Đang tạo các biểu đồ phân tích đặc trưng...")
    
    # Đặc trưng số (Numerical Features) vs Mục tiêu (Target)
    for col in config.NUMERICAL_COLUMNS:
        if col in df.columns:
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=df, x=col, y=config.TARGET_COLUMN, alpha=0.5)
            plt.title(f'{col} so với {config.TARGET_COLUMN}')
            plt.tight_layout()
            safe_col_name = col.replace(" ", "_").lower()
            plot_path = os.path.join(config.FIGURES_DIR, f"scatter_{safe_col_name}.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"Đã lưu biểu đồ phân tán: {plot_path}")

    # Đặc trưng phân loại (Categorical Features) vs Mục tiêu (Target)
    for col in config.CATEGORICAL_COLUMNS:
        if col in df.columns:
            plt.figure(figsize=(8, 6))
            sns.boxplot(data=df, x=col, y=config.TARGET_COLUMN)
            plt.title(f'{col} so với {config.TARGET_COLUMN}')
            plt.tight_layout()
            safe_col_name = col.replace(" ", "_").lower()
            plot_path = os.path.join(config.FIGURES_DIR, f"boxplot_{safe_col_name}.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"Đã lưu biểu đồ hộp: {plot_path}")
            
    print("Hoàn thành EDA.")
```
*   **Mục đích:** Thực hiện phân tích khám phá dữ liệu để hiểu rõ hơn về tập dữ liệu, các mối quan hệ giữa các biến và phân phối dữ liệu.
*   **Đầu vào:** Một DataFrame `df` (thường là dữ liệu thô hoặc đã làm sạch).
*   **Các bước thực hiện:**
    1.  **Thống kê mô tả (`df.describe()`):**
        *   Tạo bảng thống kê các thông số cơ bản (mean, std, min, max, quartiles) cho các cột số.
        *   Lưu kết quả vào file CSV trong thư mục `eda/`.
        *   **Ý nghĩa:** Cung cấp cái nhìn tổng quan về phân phối và các giá trị trung tâm, độ trải của dữ liệu.
    2.  **Biểu đồ nhiệt tương quan (Correlation Heatmap):**
        *   Tính ma trận tương quan giữa các đặc trưng số.
        *   Vẽ biểu đồ nhiệt bằng `seaborn.heatmap()`.
        *   `annot=True`: Hiển thị giá trị tương quan trên biểu đồ.
        *   `cmap='coolwarm'`: Sử dụng bảng màu để dễ nhìn.
        *   `plt.tight_layout()`: Tự động điều chỉnh khoảng cách các yếu tố trên biểu đồ để tránh bị cắt xén, đặc biệt là văn bản.
        *   Lưu biểu đồ vào `eda/figures/correlation_heatmap.png`.
        *   **Ý nghĩa:** Giúp xác định các cặp biến có mối quan hệ tuyến tính mạnh mẽ.
    3.  **Phân phối biến mục tiêu (`config.TARGET_COLUMN`):**
        *   Vẽ biểu đồ tần suất (`sns.histplot`) của biến mục tiêu.
        *   `kde=True`: Vẽ thêm đường ước tính mật độ kernel để xem hình dạng phân phối.
        *   Lưu biểu đồ vào `eda/figures/target_distribution.png`.
        *   **Ý nghĩa:** Hiểu được phân phối của biến chúng ta muốn dự đoán, giúp phát hiện dữ liệu lệch (skewed) hoặc các giá trị ngoại lai.
    4.  **Phân tích đặc trưng (Feature Analysis - Biểu đồ riêng biệt):**
        *   Thay vì một `pairplot` tổng thể, các biểu đồ riêng lẻ được tạo để tập trung vào mối quan hệ của từng đặc trưng với biến mục tiêu.
        *   **Với đặc trưng số:**
            *   Sử dụng `sns.scatterplot()` để vẽ biểu đồ phân tán của từng cột số trong `config.NUMERICAL_COLUMNS` với `config.TARGET_COLUMN`.
            *   Lưu từng biểu đồ thành file PNG riêng biệt (ví dụ: `scatter_hours_studied.png`).
            *   **Ý nghĩa:** Giúp hình dung mối quan hệ tuyến tính tiềm năng hoặc các mẫu dữ liệu giữa các đặc trưng số và biến mục tiêu.
        *   **Với đặc trưng phân loại:**
            *   Sử dụng `sns.boxplot()` để vẽ biểu đồ hộp của từng cột phân loại trong `config.CATEGORICAL_COLUMNS` với `config.TARGET_COLUMN`.
            *   Lưu từng biểu đồ thành file PNG riêng biệt (ví dụ: `boxplot_extracurricular_activities.png`).
            *   **Ý nghĩa:** Giúp so sánh phân phối của biến mục tiêu giữa các nhóm khác nhau của biến phân loại.
*   **Đầu ra:** Các file CSV thống kê và các file hình ảnh biểu đồ trong thư mục `eda/` và `eda/figures/`.
