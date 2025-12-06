# Cách Tính Gini Impurity cho thuộc tính `Weather` (với `Decision`)

Việc tính Gini cho một thuộc tính phân loại có nhiều hơn hai giá trị như `Weather` (với các giá trị `sunny`, `windy`, `rainy`) đòi hỏi phải xem xét tất cả các cách chia nhị phân (binary split) có thể có. Thuật toán CART sẽ chọn cách chia mang lại Gini có trọng số thấp nhất.

Dưới đây là chi tiết quá trình tính toán cho từng cách chia:

---

## 1. Tập dữ liệu gốc và phân bố `Decision`

- **Tổng số mẫu:** 10
- **Phân bố `Decision`:**
    - `cinema`: 6
    - `tennis`: 2
    - `stayin`: 1
    - `shopping`: 1

---

## 2. Tính Gini có trọng số cho từng cách chia nhị phân của `Weather`

### Cách chia 1: `{sunny}` vs. `{windy, rainy}`

#### a. Nhánh Trái: `Weather = sunny` (3 mẫu)
- **Các mẫu:** (w1, w2, w10)
- **Phân bố `Decision`:** `{1 cinema, 2 tennis}`
- **Tính Gini:**
    Gini(sunny) = 1 - [ (1/3)² + (2/3)² ]
                = 1 - [ 0.111 + 0.444 ]
                = **0.445**

#### b. Nhánh Phải: `Weather` là `{windy, rainy}` (7 mẫu)
- **Các mẫu:** (w3, w4, w5, w6, w7, w8, w9)
- **Phân bố `Decision`:** `{5 cinema, 1 stayin, 1 shopping}`
- **Tính Gini:**
    Gini({windy,rainy}) = 1 - [ (5/7)² + (1/7)² + (1/7)² ]
                        = 1 - [ 0.510 + 0.020 + 0.020 ]
                        = **0.450**

#### c. Gini có trọng số cho cách chia này
- **Công thức:** (Số mẫu nhánh trái / Tổng số mẫu) * Gini(nhánh trái) + (Số mẫu nhánh phải / Tổng số mẫu) * Gini(nhánh phải)
- **Tính toán:** (3/10) * 0.445 + (7/10) * 0.450
            = 0.1335 + 0.315
            = **0.4485**

---

### Cách chia 2: `{windy}` vs. `{sunny, rainy}`

#### a. Nhánh Trái: `Weather = windy` (4 mẫu)
- **Các mẫu:** (w3, w7, w8, w9)
- **Phân bố `Decision`:** `{3 cinema, 1 shopping}`
- **Tính Gini:**
    Gini(windy) = 1 - [ (3/4)² + (1/4)² ]
                = 1 - [ 0.5625 + 0.0625 ]
                = **0.375**

#### b. Nhánh Phải: `Weather` là `{sunny, rainy}` (6 mẫu)
- **Các mẫu:** (w1, w2, w4, w5, w6, w10)
- **Phân bố `Decision`:** `{3 cinema, 2 tennis, 1 stayin}`
- **Tính Gini:**
    Gini({sunny,rainy}) = 1 - [ (3/6)² + (2/6)² + (1/6)² ]
                        = 1 - [ 0.250 + 0.111 + 0.028 ]
                        = **0.611**

#### c. Gini có trọng số cho cách chia này
- **Tính toán:** (4/10) * 0.375 + (6/10) * 0.611
            = 0.150 + 0.3666
            = **0.5166**

---

### Cách chia 3: `{rainy}` vs. `{sunny, windy}`

#### a. Nhánh Trái: `Weather = rainy` (3 mẫu)
- **Các mẫu:** (w4, w5, w6)
- **Phân bố `Decision`:** `{2 cinema, 1 stayin}`
- **Tính Gini:**
    Gini(rainy) = 1 - [ (2/3)² + (1/3)² ]
                = 1 - [ 0.444 + 0.111 ]
                = **0.445**

#### b. Nhánh Phải: `Weather` là `{sunny, windy}` (7 mẫu)
- **Các mẫu:** (w1, w2, w3, w7, w8, w9, w10)
- **Phân bố `Decision`:** `{4 cinema, 2 tennis, 1 shopping}`
- **Tính Gini:**
    Gini({sunny,windy}) = 1 - [ (4/7)² + (2/7)² + (1/7)² ]
                        = 1 - [ 0.326 + 0.082 + 0.020 ]
                        = **0.572**

#### c. Gini có trọng số cho cách chia này
- **Tính toán:** (3/10) * 0.445 + (7/10) * 0.572
            = 0.1335 + 0.4004
            = **0.5339**

---

## 3. Tổng kết và lựa chọn cách chia tốt nhất cho `Weather`

| Cách chia nhị phân (`Weather`) | Gini có trọng số |
|--------------------------------|------------------|
| `{sunny}` vs. `{windy, rainy}` | **0.4485**       |
| `{windy}` vs. `{sunny, rainy}` | 0.5166           |
| `{rainy}` vs. `{sunny, windy}` | 0.5339           |

Cách chia **`{sunny}` vs. `{windy, rainy}`** có Gini có trọng số thấp nhất (0.4485), do đó đây là cách chia tốt nhất cho thuộc tính `Weather` tại nút gốc.
