import argparse
import sys
import os
import pandas as pd
import joblib
import config
from handlers import data_utils, eda_extract, model_eval
from train import train

def run_pipeline(override=False):
    print("--- Kiểm tra trạng thái Pipeline ---")
    dirty = False
    
    # 1. Làm sạch dữ liệu
    if override or not os.path.exists(config.CLEANED_DATA_PATH):
        print("Đang chạy Làm sạch dữ liệu...")
        if not os.path.exists(config.RAW_DATA_PATH):
             print(f"Lỗi: Không tìm thấy dữ liệu thô tại {config.RAW_DATA_PATH}")
             sys.exit(1)
        
        df_raw = pd.read_csv(config.RAW_DATA_PATH)
        df_clean = data_utils.data_cleaning(df_raw)
        df_clean.to_csv(config.CLEANED_DATA_PATH, index=False)
        print(f"Dữ liệu đã làm sạch được lưu tại {config.CLEANED_DATA_PATH}")
        dirty = True
    else:
        print("Dữ liệu đã làm sạch đã tồn tại. Bỏ qua bước làm sạch.")

    # 2. Chuyển đổi dữ liệu
    if override or dirty or not os.path.exists(config.TRANFORMED_DATA_PATH):
        print("Đang chạy Chuyển đổi dữ liệu...")
        df_clean = pd.read_csv(config.CLEANED_DATA_PATH)
        df_trans = data_utils.data_transform(df_clean)
        df_trans.to_csv(config.TRANFORMED_DATA_PATH, index=False)
        print(f"Dữ liệu đã chuyển đổi được lưu tại {config.TRANFORMED_DATA_PATH}")
        dirty = True
    else:
        print("Dữ liệu đã chuyển đổi đã tồn tại. Bỏ qua bước chuyển đổi.")

    # 3. Huấn luyện mô hình
    model_path = os.path.join(config.MODEL_DIR, "linear_regression_model.pkl")
    if override or dirty or not os.path.exists(model_path):
        print("Đang chạy Huấn luyện mô hình...")
        train.train_model()
        dirty = True
    else:
        print("Mô hình đã tồn tại. Bỏ qua bước huấn luyện.")

def perform_evaluation():
    model_path = os.path.join(config.MODEL_DIR, "linear_regression_model.pkl")
    X_test_path = os.path.join(config.PROCESSED_DATA_DIR, "X_test.csv")
    y_test_path = os.path.join(config.PROCESSED_DATA_DIR, "y_test.csv")
    
    if not os.path.exists(model_path) or not os.path.exists(X_test_path) or not os.path.exists(y_test_path):
        raise FileNotFoundError("Mô hình hoặc Dữ liệu kiểm tra bị thiếu.")
        
    model = joblib.load(model_path)
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)
    
    model_eval.evaluate_model(model, X_test, y_test)

def main():
    parser = argparse.ArgumentParser(description="Pipeline Hồi quy Tuyến tính")
    parser.add_argument('action', nargs='?', choices=['eval'], help='Hành động thực hiện (ví dụ: eval)')
    parser.add_argument('-eda', action='store_true', help='Thực hiện Phân tích Khám phá Dữ liệu (EDA)')
    parser.add_argument('-override', action='store_true', help='Ghi đè kết quả hiện có và chạy lại pipeline')
    
    args = parser.parse_args()
    
    # 1. Xử lý EDA
    if args.eda:
        if os.path.exists(config.RAW_DATA_PATH):
            df = pd.read_csv(config.RAW_DATA_PATH)
            eda_extract.perform_eda(df)
        else:
             print(f"Không tìm thấy dữ liệu thô tại {config.RAW_DATA_PATH} để thực hiện EDA.")

    # 2. Xử lý lệnh 'eval'
    if args.action == 'eval':
        try:
            perform_evaluation()
        except Exception as e:
            print(f"Đánh giá thất bại: {e}")
            print("Đang thử chạy pipeline chuẩn...")
            try:
                run_pipeline(override=False)
                perform_evaluation()
            except Exception as e2:
                print(f"Pipeline chuẩn/Đánh giá thất bại: {e2}")
                print("Đang thử chạy pipeline với chế độ ghi đè (override)...")
                try:
                    run_pipeline(override=True)
                    perform_evaluation()
                except Exception as e3:
                    print(f"Lỗi nghiêm trọng: Không thể thực hiện đánh giá. {e3}")
                    sys.exit(1)
    
    # 3. Xử lý thực thi Pipeline mặc định
    else:
        should_run_pipeline = False
        if args.override:
            should_run_pipeline = True
        elif not args.eda: # Không override, không EDA, không 'eval'
            should_run_pipeline = True
        
        if should_run_pipeline:
            run_pipeline(override=args.override)
            perform_evaluation()

if __name__ == "__main__":
    main()
