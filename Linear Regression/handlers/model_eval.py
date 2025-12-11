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
        f.write(f"--- Đánh giá Mô hình (Tập {dataset_name}) ---")
        f.write(f"R-squared (Độ chính xác): {r2:.4f}")
        f.write(f"Mean Squared Error (MSE): {mse:.4f}")
        f.write(f"Mean Absolute Error (MAE): {mae:.4f}")
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
