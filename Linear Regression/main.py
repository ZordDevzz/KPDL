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

    # 2. Huấn luyện mô hình (bao gồm chuyển đổi dữ liệu và lưu transformers)
    model_path = os.path.join(config.MODEL_DIR, "linear_regression_model.pkl")
    if override or dirty or not os.path.exists(model_path) or \
       not os.path.exists(config.SCALER_PATH) or not os.path.exists(config.LABEL_ENCODER_PATH):
        print("Đang chạy Huấn luyện mô hình (bao gồm chuyển đổi dữ liệu và lưu transformers)...")
        train.train_model()
        dirty = True
    else:
        print("Mô hình và Transformers đã tồn tại. Bỏ qua bước huấn luyện.")

def perform_evaluation():
    model_path = os.path.join(config.MODEL_DIR, "linear_regression_model.pkl")
    X_test_path = os.path.join(config.PROCESSED_DATA_DIR, "X_test.csv")
    y_test_path = os.path.join(config.PROCESSED_DATA_DIR, "y_test.csv")
    
    if not os.path.exists(model_path) or not os.path.exists(X_test_path) or not os.path.exists(y_test_path):
        raise FileNotFoundError("Mô hình hoặc Dữ liệu kiểm tra bị thiếu. Hãy chạy pipeline trước.")
        
    model = joblib.load(model_path)
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)
    
    model_eval.evaluate_model(model, X_test, y_test)

def perform_prediction(input_csv_path):
    print(f"--- Đang thực hiện dự đoán trên tệp: {input_csv_path} ---")
    
    # Kiểm tra sự tồn tại của mô hình và transformers
    model_path = os.path.join(config.MODEL_DIR, "linear_regression_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Mô hình không tìm thấy tại {model_path}. Hãy huấn luyện mô hình trước.")
    if not os.path.exists(config.SCALER_PATH) or not os.path.exists(config.LABEL_ENCODER_PATH):
        raise FileNotFoundError(f"Transformers (Scaler hoặc LabelEncoder) không tìm thấy. Hãy huấn luyện mô hình trước.")
        
    # Tải mô hình và transformers
    model = joblib.load(model_path)
    scaler, label_encoder = data_utils.load_transformers()
    
    # Tải dữ liệu đầu vào
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Tệp đầu vào không tìm thấy tại: {input_csv_path}")
        
    df_input = pd.read_csv(input_csv_path)
    print(f"Đã tải {len(df_input)} bản ghi từ {input_csv_path}.")

    # Xử lý làm sạch và chuyển đổi dữ liệu đầu vào
    df_cleaned = data_utils.data_cleaning(df_input.copy())
    # Sử dụng các transformer đã được khớp từ quá trình huấn luyện
    df_processed, _, _ = data_utils.data_transform(df_cleaned, scaler=scaler, label_encoder=label_encoder, is_predict=True)
    
    # Đảm bảo các cột đầu vào khớp với các cột đã huấn luyện
    # Cần phải lấy danh sách các cột đã huấn luyện.
    # Hiện tại, chúng ta sẽ giả định các cột đầu vào khớp hoàn toàn.
    # Trong một hệ thống mạnh mẽ hơn, chúng ta sẽ tải X_train.columns hoặc tương tự.
    
    # Để đơn giản, lấy các cột từ dữ liệu đã chuyển đổi (trừ cột mục tiêu nếu có)
    # Nếu df_processed chứa cột mục tiêu, ta cần loại bỏ nó trước khi dự đoán
    predict_df = df_processed.drop(columns=[config.TARGET_COLUMN], errors='ignore')
    
    # Thực hiện dự đoán
    predictions = model.predict(predict_df)
    
    # Thêm dự đoán vào DataFrame gốc để hiển thị
    df_input['Predicted_Performance_Index'] = predictions
    
    print("\n--- Kết quả dự đoán ---")
    print(df_input[[col for col in df_input.columns if col not in [config.TARGET_COLUMN]] + ['Predicted_Performance_Index']].head())
    
    output_path = os.path.join(config.PROCESSED_DATA_DIR, "predictions.csv")
    df_input.to_csv(output_path, index=False)
    print(f"\nKết quả dự đoán đã được lưu vào: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Pipeline Hồi quy Tuyến tính")
    parser.add_argument('action', nargs='?', choices=['eval'], help='Hành động thực hiện (ví dụ: eval)')
    parser.add_argument('-eda', action='store_true', help='Thực hiện Phân tích Khám phá Dữ liệu (EDA)')
    parser.add_argument('-override', action='store_true', help='Ghi đè kết quả hiện có và chạy lại pipeline')
    parser.add_argument('-predict', type=str, help='Đường dẫn đến tệp CSV đầu vào để dự đoán')
    
    args = parser.parse_args()

    # 1. Xử lý lệnh 'predict'
    if args.predict:
        try:
            perform_prediction(args.predict)
        except FileNotFoundError as e:
            print(f"Lỗi: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Đã xảy ra lỗi trong quá trình dự đoán: {e}")
            sys.exit(1)
        return # Thoát sau khi dự đoán

    # 2. Xử lý EDA
    if args.eda:
        if os.path.exists(config.RAW_DATA_PATH):
            df = pd.read_csv(config.RAW_DATA_PATH)
            eda_extract.perform_eda(df)
        else:
             print(f"Không tìm thấy dữ liệu thô tại {config.RAW_DATA_PATH} để thực hiện EDA.")
        return # Thoát sau khi EDA

    # 3. Xử lý lệnh 'eval'
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
        return # Thoát sau khi eval
    
    # 4. Xử lý thực thi Pipeline mặc định
    should_run_pipeline = False
    if args.override:
        should_run_pipeline = True
    elif not args.eda: # Không override, không EDA, không 'eval', không 'predict'
        should_run_pipeline = True
    
    if should_run_pipeline:
        run_pipeline(override=args.override)
        perform_evaluation()

if __name__ == "__main__":
    main()
