# Phân Tích Cây Quyết Định CART (Theo phương pháp rút gọn)

Tài liệu này trình bày lại quá trình xây dựng cây quyết định cho `data.csv` bằng thuật toán CART, áp dụng phương pháp lựa chọn thuộc tính rút gọn:
1.  **Đánh giá thuộc tính:** Sử dụng Gini đa nhánh (multi-way split) để tính "điểm chất lượng" cho mỗi thuộc tính.
2.  **Phân nhánh:** Chọn thuộc tính có điểm Gini thấp nhất để thực hiện **phân nhánh nhị phân (binary split)**.

---

## 1. Tìm Nút Gốc (Root Node)

### Bước 1.1: Đánh giá chất lượng của các thuộc tính

Ta tính "điểm Gini" cho mỗi thuộc tính để so sánh.

#### a. Thuộc tính `Parrent` (2 giá trị)
- Gini có trọng số = **0.36**
*(Chi tiết trong file output.txt, vì là thuộc tính nhị phân nên cách tính không đổi)*

#### b. Thuộc tính `Money` (2 giá trị)
- Gini có trọng số = **0.486**
*(Chi tiết trong file output.txt, cách tính không đổi)*

#### c. Thuộc tính `Weather` (3 giá trị) - *Tính theo Gini đa nhánh*
- Gini(sunny) [3 mẫu] = 0.444
- Gini(windy) [4 mẫu] = 0.375
- Gini(rainy) [3 mẫu] = 0.444
- **Điểm Gini đa nhánh (`Weather`)** = (3/10)*0.444 + (4/10)*0.375 + (3/10)*0.444 = **0.4164**

#### Bảng tổng kết đánh giá thuộc tính

| Thuộc tính | "Điểm Gini" (Chất lượng) |
|------------|---------------------------|
| **Parrent**  | **0.36**                  |
| Money      | 0.486                     |
| Weather    | 0.4164                    |

=> `Parrent` là thuộc tính tốt nhất để phân nhánh vì có điểm Gini thấp nhất.

### Bước 1.2: Thực hiện phân nhánh nhị phân cho `Parrent`

- Thuộc tính được chọn là `Parrent`.
- Vì `Parrent` chỉ có 2 giá trị (`yes`, `no`), phép chia nhị phân rất tự nhiên: một nhánh cho `yes` và một nhánh cho `no`.

**Nút gốc của cây được xác định là `Parrent`.**

---

## 2. Phát triển các nhánh từ Nút Gốc

- **Nhánh 1: `Parrent` = 'yes'**
  - Tập dữ liệu này chứa 5 mẫu, tất cả đều có `Decision` là 'cinema'.
  - => Đây là một **nút lá (leaf node)** thuần khiết với kết quả là **`cinema`**.

- **Nhánh 2: `Parrent` = 'no'**
  - Tập dữ liệu con này có 5 mẫu và Gini là 0.72. Cần phân nhánh tiếp.
  - Các thuộc tính còn lại: `Weather`, `Money`.

### Bước 2.1: Đánh giá chất lượng các thuộc tính cho nút con `Parrent = 'no'`

#### a. Thuộc tính `Money` (2 giá trị)
- Trên tập 5 mẫu này, Gini có trọng số của `Money` là **0.5**.
*(Chi tiết: 4 mẫu 'rich' có Gini 0.625, 1 mẫu 'poor' có Gini 0. Trọng số: (4/5)*0.625 + (1/5)*0 = 0.5)*

#### b. Thuộc tính `Weather` (3 giá trị) - *Tính theo Gini đa nhánh*
- Trên tập 5 mẫu này:
  - Gini(sunny) [2 mẫu: tennis, tennis] = 0
  - Gini(rainy) [1 mẫu: stayin] = 0
  - Gini(windy) [2 mẫu: cinema, shopping] = 0.5
- **Điểm Gini đa nhánh (`Weather`)** = (2/5)*0 + (1/5)*0 + (2/5)*0.5 = **0.2**

#### Bảng tổng kết đánh giá
| Thuộc tính | "Điểm Gini" (Chất lượng) |
|------------|---------------------------|
| Money      | 0.5                       |
| **Weather**  | **0.2**                   |

=> `Weather` là thuộc tính tốt nhất để phân nhánh tiếp theo.

### Bước 2.2: Thực hiện phân nhánh nhị phân cho `Weather`

- Thuộc tính được chọn là `Weather`.
- Bây giờ, ta phải tìm cách **chia đôi (binary split)** tốt nhất cho `Weather` trên tập dữ liệu con này.
- Các cách chia có thể:
  - `{sunny}` vs `{rainy, windy}` -> Gini trọng số = **0.4**
  - `{rainy}` vs `{sunny, windy}` -> Gini trọng số = 0.5
  - `{windy}` vs `{sunny, rainy}` -> Gini trọng số = 0.466
- => Phép chia nhị phân tốt nhất là **`{sunny}` vs `{rainy, windy}`**.

---

## 3. Phát triển các nhánh từ nút `Weather`

- **Nhánh 2.1: `Weather` = 'sunny'** (trong nhánh `Parrent` = 'no')
  - Có 2 mẫu, đều là `tennis`.
  - => Đây là **nút lá** thuần khiết với kết quả **`tennis`**.

- **Nhánh 2.2: `Weather` thuộc `{rainy, windy}`** (trong nhánh `Parrent` = 'no')
  - Có 3 mẫu: {stayin, cinema, shopping}. Cần chia tiếp.
  - Thuộc tính còn lại duy nhất là `Money`.
  - Phân nhánh theo `Money`:
    - **Nhánh 2.2.1: `Money` = 'poor'** (1 mẫu) => **Nút lá** thuần khiết, kết quả là **`cinema`**.
    - **Nhánh 2.2.2: `Money` = 'rich'** (2 mẫu) => Hết thuộc tính để chia. **Nút lá không thuần khiết**. Kết quả dự đoán là `stayin` hoặc `shopping`.

---

## 4. Kết quả: Cây Quyết Định Hoàn Chỉnh

Cấu trúc cây quyết định cuối cùng không thay đổi, nhưng cách thức lựa chọn thuộc tính ở mỗi bước đã tuân theo phương pháp rút gọn.

1.  **Hỏi: `Parrent` == 'yes'?**
    *   **Đúng:** => **`cinema`**
    *   **Sai:** => Đi đến câu hỏi 2.

2.  **Hỏi: `Weather` == 'sunny'?**
    *   **Đúng:** => **`tennis`**
    *   **Sai** (`Weather` là 'rainy' hoặc 'windy'): => Đi đến câu hỏi 3.

3.  **Hỏi: `Money` == 'poor'?**
    *   **Đúng:** => **`cinema`**
    *   **Sai** (`Money` là 'rich'): => **`stayin` hoặc `shopping`**
