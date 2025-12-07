# Phân tích chi tiết thuật toán Naive Bayes
Đây là tài liệu hướng dẫn từng bước về cách hoạt động của thuật toán Naive Bayes cho mục đích giáo dục.

Naive Bayes là một thuật toán phân loại dựa trên Định lý Bayes với giả định 'ngây thơ' (naive) rằng các thuộc tính là độc lập với nhau khi biết lớp quyết định. Mặc dù giả định này hiếm khi đúng trong thực tế, thuật toán vẫn hoạt động hiệu quả trong nhiều trường hợp.

## 1. Dữ liệu và Phân tích Tần suất
- **Tập dữ liệu:** `data.csv`
- **Mẫu mới cần phân loại:** `{'humid': 'normal', 'weather': 'sunny', 'wind': 'weak'}`

## 2. Giai đoạn 'Huấn luyện': Tính các xác suất
Trong Naive Bayes, 'huấn luyện' đơn giản là tính toán các xác suất từ tập dữ liệu đã cho.

### a) Tính Xác suất Tiên nghiệm P(Class) - (Prior Probabilities)
Đây là xác suất xuất hiện của mỗi lớp trong toàn bộ tập dữ liệu.

- **P(decision=no)** = (Số lần 'no' xuất hiện) / (Tổng số mẫu) = 4/6 = **0.667**
- **P(decision=yes)** = (Số lần 'yes' xuất hiện) / (Tổng số mẫu) = 2/6 = **0.333**

### b) Tính Xác suất có điều kiện P(Thuộc tính | Class) - (Conditional Probabilities)
Đây là xác suất của một giá trị thuộc tính cụ thể, biết trước lớp của nó. Chúng ta sử dụng kỹ thuật **Làm mịn Laplace (add-1)** để tránh xác suất bằng 0 khi một giá trị không xuất hiện trong một lớp nào đó.

Công thức (với làm mịn Laplace): `P(Value | Class) = (Số lần Value xuất hiện trong Class + 1) / (Tổng số mẫu của Class + Số lượng giá trị khác nhau của thuộc tính)`

#### Phân tích thuộc tính: `humid`
- **P(humid=high | decision=no)** = (2 + 1) / (4 + 2) = **0.500**
- **P(humid=normal | decision=no)** = (2 + 1) / (4 + 2) = **0.500**
- **P(humid=high | decision=yes)** = (1 + 1) / (2 + 2) = **0.500**
- **P(humid=normal | decision=yes)** = (1 + 1) / (2 + 2) = **0.500**
#### Phân tích thuộc tính: `weather`
- **P(weather=overcast | decision=no)** = (0 + 1) / (4 + 3) = **0.143**
- **P(weather=rainy | decision=no)** = (3 + 1) / (4 + 3) = **0.571**
- **P(weather=sunny | decision=no)** = (1 + 1) / (4 + 3) = **0.286**
- **P(weather=overcast | decision=yes)** = (1 + 1) / (2 + 3) = **0.400**
- **P(weather=rainy | decision=yes)** = (0 + 1) / (2 + 3) = **0.200**
- **P(weather=sunny | decision=yes)** = (1 + 1) / (2 + 3) = **0.400**
#### Phân tích thuộc tính: `wind`
- **P(wind=strong | decision=no)** = (2 + 1) / (4 + 2) = **0.500**
- **P(wind=weak | decision=no)** = (2 + 1) / (4 + 2) = **0.500**
- **P(wind=strong | decision=yes)** = (0 + 1) / (2 + 2) = **0.250**
- **P(wind=weak | decision=yes)** = (2 + 1) / (2 + 2) = **0.750**

## 3. Giai đoạn Dự đoán
Áp dụng các xác suất đã tính để dự đoán lớp cho mẫu mới: `{'humid': 'normal', 'weather': 'sunny', 'wind': 'weak'}`

### Công thức dự đoán của Naive Bayes:
Tìm lớp `C` để tối đa hóa biểu thức sau:
`P(C | Features) ∝ P(C) * P(Feature_1 | C) * P(Feature_2 | C) * ...`

### Tính toán cho từng lớp:
#### Lớp (Class): **no**
- **Tính toán:** 0.667 (P(decision=no)) * 0.500 (P(humid=normal|no)) * 0.286 (P(weather=sunny|no)) * 0.500 (P(wind=weak|no))
- **Kết quả (chưa chuẩn hóa):** 0.04762

#### Lớp (Class): **yes**
- **Tính toán:** 0.333 (P(decision=yes)) * 0.500 (P(humid=normal|yes)) * 0.400 (P(weather=sunny|yes)) * 0.750 (P(wind=weak|yes))
- **Kết quả (chưa chuẩn hóa):** 0.05000

## 4. Tổng kết và Đưa ra Quyết định
So sánh các kết quả tính toán để tìm ra lớp có xác suất cao nhất.

| Lớp (Class) | Điểm xác suất (Posterior Score) |
|---|---|
| no | 0.04762 |
| yes | 0.05000 |

**=> Kết luận:** Lớp có điểm xác suất cao nhất là **'yes'**. Đây là dự đoán cuối cùng.

### Phụ lục: Chuẩn hóa Xác suất (Optional)
Để chuyển các điểm thành xác suất có tổng bằng 1, chúng ta có thể chuẩn hóa chúng:

| Lớp (Class) | Xác suất đã chuẩn hóa P(Class | Features) |
|---|---|
| no | 0.488 |
| yes | 0.512 |