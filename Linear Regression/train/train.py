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