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
    # Tăng kích thước hình và sử dụng tight_layout để tránh cắt chữ
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
            # Làm sạch tên file
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
