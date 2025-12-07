# Tổng hợp tài liệu tính toán các thuật toán Machine Learning

Tài liệu này tóm tắt các ví dụ tính toán cho 3 thuật toán: CART, Naive Bayes và KNN, dựa trên các tài liệu viết tay. Các lỗi tính toán đã được xác minh và sửa chữa.

## 1. Thuật toán CART (Classification and Regression Trees)

Thuật toán CART xây dựng cây quyết định dựa trên chỉ số Gini, một độ đo về sự "tinh khiết" của một nút. Thuộc tính có chỉ số Gini thấp nhất được chọn để phân nhánh.

### Ví dụ

**Bài toán:** Dựa vào các yếu tố `Weather`, `Parents`, `Money` để dự đoán `Decision`.

**Dữ liệu ban đầu:**

| Weather | Parents | Money | Decision |
| :--- | :--- | :--- | :--- |
| Sunny | Yes | Rich | Cinema |
| Sunny | No | Rich | Tennis |
| Windy | Yes | Rich | Cinema |
| Rainy | Yes | Poor | Cinema |
| Rainy | No | Rich | Stay in |
| Rainy | Yes | Poor | Cinema |
| Windy | No | Poor | Cinema |
| Windy | No | Rich | Shopping |
| Windy | Yes | Rich | Cinema |
| Sunny | No | Rich | Tennis |

**Bước 1: Chọn thuộc tính gốc**

1.  **Tính Gini Index cho toàn bộ tập dữ liệu (D):**
    *   Quyết định: Cinema (6), Tennis (2), Stay in (1), Shopping (1). Tổng: 10.
    *   `Gini(D) = 1 - [(6/10)² + (2/10)² + (1/10)² + (1/10)²] = 1 - (0.36 + 0.04 + 0.01 + 0.01) = 0.58`
    *   *Ghi chú: Phép tính trong tài liệu gốc là đúng.*

2.  **Tính Gini cho từng thuộc tính:**
    *   **Gini(Weather):**
        *   Sunny (3): 1 Cinema, 2 Tennis
        *   Windy (4): 3 Cinema, 1 Shopping
        *   Rainy (3): 2 Cinema, 1 Stay in
        *   `Gini(Weather) = (3/10) * [1 - (1/3)² - (2/3)²] + (4/10) * [1 - (3/4)² - (1/4)²] + (3/10) * [1 - (2/3)² - (1/3)²]`
        *   `= 0.3 * 0.444 + 0.4 * 0.375 + 0.3 * 0.444 = 0.1332 + 0.15 + 0.1332 = 0.4164`
        *   *Ghi chú: Phép tính trong tài liệu gốc (0.42) có sai số nhỏ.*

    *   **Gini(Parents):**
        *   Yes (5): 5 Cinema
        *   No (5): 2 Tennis, 1 Stay in, 1 Cinema, 1 Shopping
        *   `Gini(Parents) = (5/10) * [1 - (5/5)²] + (5/10) * [1 - (2/5)² - (1/5)² - (1/5)² - (1/5)²]`
        *   `= 0.5 * 0 + 0.5 * [1 - 0.16 - 0.04 - 0.04 - 0.04] = 0.5 * 0.72 = 0.36`
        *   *Ghi chú: Phép tính trong tài liệu gốc là đúng.*

    *   **Gini(Money):**
        *   Rich (7): 3 Cinema, 2 Tennis, 1 Shopping, 1 Stay in
        *   Poor (3): 3 Cinema
        *   `Gini(Money) = (7/10) * [1 - (3/7)² - (2/7)² - (1/7)² - (1/7)²] + (3/10) * [1 - (3/3)²]`
        *   `= 0.7 * [1 - 0.183 - 0.081 - 0.02 - 0.02] + 0.3 * 0 = 0.7 * 0.696 = 0.487`
        *   *Ghi chú: Phép tính trong tài liệu gốc (0.48) có sai số nhỏ.*

3.  **Kết luận:** `Gini(Parents) = 0.36` là nhỏ nhất. **Chọn `Parents` làm thuộc tính gốc.**

---

## 2. Thuật toán Naive Bayes

Naive Bayes là một thuật toán phân loại dựa trên định lý Bayes với giả định "ngây thơ" rằng các thuộc tính là độc lập với nhau.

### Ví dụ

**Bài toán:** Dự đoán `Play Golf` dựa trên `Outlook`, `Temperature`, `Humidity`, `Windy`.

**Trường hợp cần dự đoán (X):** `Outlook = Sunny`, `Temperature = Hot`, `Humidity = Normal`, `Windy = False`.

**Dữ liệu:**
*   Tổng số 14 bản ghi.
*   `P(Yes) = 9/14`
*   `P(No) = 5/14`

**Phân tích (đã sửa lỗi tính toán so với tài liệu gốc):**

1.  **Tính xác suất cho lớp `Yes`:**
    *   `P(Sunny|Yes) = 2/9`
    *   `P(Hot|Yes) = 2/9`
    *   `P(Normal|Yes) = 6/9`
    *   `P(False|Yes) = 6/9`
    *   `P(X|Yes) * P(Yes) = (2/9) * (2/9) * (6/9) * (6/9) * (9/14) = (48/6561) * (9/14) ≈ 0.0047`
    *   *Ghi chú: Tài liệu gốc tính sai P(Sunny|Yes) là 3/9 dẫn đến kết quả 0.021.*

2.  **Tính xác suất cho lớp `No`:**
    *   `P(Sunny|No) = 3/5`
    *   `P(Hot|No) = 2/5`
    *   `P(Normal|No) = 1/5`
    *   `P(False|No) = 2/5`
    *   `P(X|No) * P(No) = (3/5) * (2/5) * (1/5) * (2/5) * (5/14) = (12/625) * (5/14) ≈ 0.0068`
    *   *Ghi chú: Tài liệu gốc tính sai P(Sunny|No) là 2/5 dẫn đến kết quả 0.004.*

3.  **Kết luận:**
    *   Vì `0.0068 > 0.0047`, tức là `P(X|No) * P(No) > P(X|Yes) * P(Yes)`.
    *   **Dự đoán là `Play Golf = No`**.
    *   *Ghi chú: Kết luận trong tài liệu gốc là `Yes`, không chính xác do lỗi tính toán.*

---

## 3. Thuật toán KNN (K-Nearest Neighbors)

KNN phân loại một điểm dữ liệu mới bằng cách xem xét `k` hàng xóm gần nhất của nó trong không gian đặc trưng.

### Ví dụ

**Bài toán:** Dự đoán `variety` cho `ID13` (petal.length=4.4, petal.width=1.4) với `k=5`.

**Dữ liệu:** 12 mẫu hoa `Setosa` và `Versicolor`.

**Phân tích:**

1.  **Tính khoảng cách Euclidean** từ `ID13` đến tất cả các điểm khác.
2.  **Xác định 5 hàng xóm gần nhất:**

| ID | Petal Length | Petal Width | Variety | Khoảng cách đến ID13 | Rank |
| :--- | :--- | :--- | :--- | :--- | :--- |
| ID7 | 4.5 | 1.5 | Versicolor | 0.14 | 1 |
| ID11 | 4.5 | 1.3 | Versicolor | 0.14 | 1 |
| ID10 | 4.6 | 1.5 | Versicolor | 0.22 | 3 |
| ID6 | 4.7 | 1.4 | Versicolor | 0.30 | 4 |
| ID12 | 4.7 | 1.6 | Versicolor | 0.36 | 5 |
| ID9 | 4.0 | 1.3 | Versicolor | 0.41 | 6 |
| ID8 | 4.9 | 1.5 | Versicolor | 0.51 | 7 |
| ... | ... | ... | ... | ... | ... |

*Ghi chú: Có sự trùng lặp về khoảng cách (ID7 và ID11), cả hai đều là hàng xóm gần nhất.*

3.  **Kết luận:**
    *   5 hàng xóm gần nhất là: ID7, ID11, ID10, ID6, ID12.
    *   Tất cả 5 hàng xóm này đều thuộc lớp `Versicolor`.
    *   **Dự đoán cho `ID13` là `Versicolor`**.
    *   *Ghi chú: Kết quả trong tài liệu gốc là đúng.*
