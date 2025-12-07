# Phân tích chi tiết thuật toán k-Nearest Neighbors (k-NN)
Đây là tài liệu hướng dẫn từng bước về cách hoạt động của thuật toán k-NN cho mục đích giáo dục.

k-NN là một thuật toán 'học lười' (lazy learning), nghĩa là nó không xây dựng một mô hình rõ ràng. Thay vào đó, nó lưu trữ toàn bộ tập dữ liệu huấn luyện và thực hiện dự đoán dựa trên sự tương đồng (khoảng cách) với các điểm dữ liệu mới.

## 1. Dữ liệu và Tham số
- **Tập dữ liệu:** `data.csv`
- **Giá trị k:** 3
- **Mẫu mới cần phân loại:** `{'humid': 'normal', 'weather': 'sunny', 'wind': 'weak'}`

### Dữ liệu gốc:
| weekend | humid | weather | wind | decision |
|---|---|---|---|---|
| w1 | normal | rainy | weak | no |
| w2 | normal | rainy | strong | no |
| w3 | high | rainy | weak | no |
| w4 | high | sunny | strong | no |
| w5 | normal | sunny | weak | yes |
| w6 | high | overcast | weak | yes |

## 2. Mã hóa One-Hot (One-Hot Encoding)
Vì các thuộc tính của chúng ta là dạng chữ (categorical), chúng ta cần chuyển chúng thành dạng số để có thể tính toán khoảng cách. Mã hóa One-Hot là một phương pháp phổ biến để làm việc này.

### Sơ đồ mã hóa:
- **humid**:
  - `high` -> `10`
  - `normal` -> `01`
- **weather**:
  - `overcast` -> `100`
  - `rainy` -> `010`
  - `sunny` -> `001`
- **wind**:
  - `strong` -> `10`
  - `weak` -> `01`

### Dữ liệu sau khi mã hóa:
| weekend | humid | weather | wind | decision | Vector mã hóa |
|---|---|---|---|---|---|
| normal | rainy | weak | no | `[0, 1, 0, 1, 0, 0, 1]` |
| normal | rainy | strong | no | `[0, 1, 0, 1, 0, 1, 0]` |
| high | rainy | weak | no | `[1, 0, 0, 1, 0, 0, 1]` |
| high | sunny | strong | no | `[1, 0, 0, 0, 1, 1, 0]` |
| normal | sunny | weak | yes | `[0, 1, 0, 0, 1, 0, 1]` |
| high | overcast | weak | yes | `[1, 0, 1, 0, 0, 0, 1]` |

- **Vector mã hóa cho mẫu mới:** `[0, 1, 0, 0, 1, 0, 1]`

## 3. Tính khoảng cách Euclid (Euclidean Distance)
Bây giờ, chúng ta tính khoảng cách từ mẫu mới đến TẤT CẢ các mẫu trong tập dữ liệu huấn luyện.

### Công thức:
`Distance(A, B) = sqrt( Σ(A_i - B_i)^2 )`

| ID Gốc | Vector Huấn luyện | Vector Mới | Tính toán Khoảng cách | Kết quả (Distance) | Quyết định |
|---|---|---|---|---|---|
| w1 | `[0, 1, 0, 1, 0, 0, 1]` | `[0, 1, 0, 0, 1, 0, 1]` | `sqrt((0-0)^2 + (1-1)^2 + (0-0)^2 + (1-0)^2 + (0-1)^2 + (0-0)^2 + (1-1)^2)` | **1.414** | no |
| w2 | `[0, 1, 0, 1, 0, 1, 0]` | `[0, 1, 0, 0, 1, 0, 1]` | `sqrt((0-0)^2 + (1-1)^2 + (0-0)^2 + (1-0)^2 + (0-1)^2 + (1-0)^2 + (0-1)^2)` | **2.000** | no |
| w3 | `[1, 0, 0, 1, 0, 0, 1]` | `[0, 1, 0, 0, 1, 0, 1]` | `sqrt((1-0)^2 + (0-1)^2 + (0-0)^2 + (1-0)^2 + (0-1)^2 + (0-0)^2 + (1-1)^2)` | **2.000** | no |
| w4 | `[1, 0, 0, 0, 1, 1, 0]` | `[0, 1, 0, 0, 1, 0, 1]` | `sqrt((1-0)^2 + (0-1)^2 + (0-0)^2 + (0-0)^2 + (1-1)^2 + (1-0)^2 + (0-1)^2)` | **2.000** | no |
| w5 | `[0, 1, 0, 0, 1, 0, 1]` | `[0, 1, 0, 0, 1, 0, 1]` | `sqrt((0-0)^2 + (1-1)^2 + (0-0)^2 + (0-0)^2 + (1-1)^2 + (0-0)^2 + (1-1)^2)` | **0.000** | yes |
| w6 | `[1, 0, 1, 0, 0, 0, 1]` | `[0, 1, 0, 0, 1, 0, 1]` | `sqrt((1-0)^2 + (0-1)^2 + (1-0)^2 + (0-0)^2 + (0-1)^2 + (0-0)^2 + (1-1)^2)` | **2.000** | yes |

## 4. Sắp xếp và tìm 3 láng giềng gần nhất
Sắp xếp tất cả các điểm dữ liệu theo khoảng cách tăng dần và chọn ra k điểm đầu tiên.

| ID Gốc | Quyết định | Khoảng cách | Là láng giềng? |
|---|---|---|---|
| w5 | yes | 0.000 | ✅ |
| w1 | no | 1.414 | ✅ |
| w2 | no | 2.000 | ✅ |
| w3 | no | 2.000 | ❌ |
| w4 | no | 2.000 | ❌ |
| w6 | yes | 2.000 | ❌ |

## 5. Dự đoán dựa trên bỏ phiếu đa số
Các láng giềng được chọn là: w5 (yes), w1 (no), w2 (no).

- **Các phiếu bầu (votes) từ các láng giềng:**
  - Lớp 'yes': 1 phiếu
  - Lớp 'no': 2 phiếu

**=> Kết luận:** Lớp có nhiều phiếu bầu nhất là **'no'**. Do đó, đây là dự đoán cuối cùng cho mẫu mới.