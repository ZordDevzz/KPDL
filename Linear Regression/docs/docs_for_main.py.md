# Tài liệu hướng dẫn: main.py

Tệp `main.py` là điểm khởi chạy chính (entry point) của ứng dụng. Nó kết nối tất cả các thành phần (làm sạch, chuyển đổi, huấn luyện, đánh giá) thành một quy trình làm việc thống nhất và cung cấp giao diện dòng lệnh (CLI) để người dùng tương tác.

## Chi tiết Mã nguồn

### 1. Hàm `run_pipeline(override=False)`

```python
def run_pipeline(override=False):
    # ... (code chi tiết xem trong file source)
```
*   **Mục đích:** Điều phối việc thực thi tuần tự các bước trong pipeline.
*   **Cơ chế "Dirty Checking":**
    *   Hàm này kiểm tra xem kết quả đầu ra của từng bước (ví dụ: file dữ liệu sạch, file mô hình) đã tồn tại hay chưa.
    *   Nếu file chưa tồn tại HOẶC người dùng yêu cầu `override` HOẶC một bước trước đó vừa được chạy lại (`dirty=True`), bước hiện tại sẽ được thực thi.
    *   Nếu không, nó sẽ bỏ qua để tiết kiệm thời gian.
    *   Ví dụ: Nếu dữ liệu đã được làm sạch và chuyển đổi, nhưng mô hình chưa được huấn luyện, pipeline sẽ bỏ qua 2 bước đầu và chỉ chạy bước huấn luyện.

### 2. Hàm `perform_evaluation()`

```python
def perform_evaluation():
    # ... (code chi tiết xem trong file source)
```
*   **Mục đích:** Thực hiện đánh giá mô hình độc lập.
*   **Hoạt động:**
    *   Tải mô hình (`linear_regression_model.pkl`) và dữ liệu kiểm tra (`X_test.csv`, `y_test.csv`) từ đĩa.
    *   Gọi hàm `evaluate_model` từ `handlers/model_eval.py` để tính toán độ chính xác và vẽ biểu đồ.
    *   Nếu thiếu file, nó sẽ ném lỗi `FileNotFoundError` để hàm `main` xử lý.

### 3. Hàm `main()`

```python
def main():
    parser = argparse.ArgumentParser(description="Pipeline Hồi quy Tuyến tính")
    parser.add_argument('action', nargs='?', choices=['eval'], help='Hành động thực hiện (ví dụ: eval)')
    parser.add_argument('-eda', action='store_true', help='Thực hiện Phân tích Khám phá Dữ liệu (EDA)')
    parser.add_argument('-override', action='store_true', help='Ghi đè kết quả hiện có và chạy lại pipeline')
    
    args = parser.parse_args()
    
    # ... (logic xử lý args)
```
*   **Mục đích:** Xử lý các tham số dòng lệnh và quyết định luồng thực thi của chương trình.
*   **Các tham số:**
    *   `action='eval'`: Chỉ định chế độ chỉ đánh giá.
    *   `-eda`: Bật chế độ phân tích dữ liệu.
    *   `-override`: Bắt buộc chạy lại toàn bộ pipeline.
*   **Logic xử lý lỗi & Tự động phục hồi:**
    *   Khi người dùng chạy `python main.py eval`, nếu chương trình phát hiện thiếu file mô hình hoặc dữ liệu (gây ra lỗi), nó sẽ không dừng ngay.
    *   Thay vào đó, nó sẽ tự động thử chạy `run_pipeline()` để tạo ra các file còn thiếu, sau đó thử đánh giá lại.
    *   Đây là tính năng giúp trải nghiệm người dùng mượt mà hơn, không cần phải nhớ chạy pipeline trước khi chạy đánh giá.
