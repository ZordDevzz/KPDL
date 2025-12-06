# Phân Tích Mối Quan Hệ Giữa `Weather` và `Decision`

Tài liệu này trình bày chi tiết cách tính toán Gini Impurity và xác định cách chia tốt nhất cho thuộc tính `Weather` để phân lớp `Decision`, sử dụng dữ liệu được cung cấp trong `weather_decision.csv`.

---

## 1. Dữ liệu từ `weather_decision.csv`

Dữ liệu đầu vào đã được tổng hợp dưới dạng bảng tần suất:

| weather | cinema | tennis | stayin | shopping | total |
|---------|--------|--------|--------|----------|-------|
| sunny   | 1      | 2      | 0      | 0        | 3     |
| windy   | 3      | 0      | 0      | 1        | 4     |
| rainy   | 2      | 0      | 1      | 0        | 3     |
| **total** | **6**    | **2**    | **1**    | **1**      | **10**  |

---

## 2. Tính Gini Impurity của Nút Gốc (Toàn bộ dữ liệu)

Gini Impurity của nút gốc được tính dựa trên tổng phân bố của các lớp `Decision`:
- `cinema`: 6/10 = 0.6
- `tennis`: 2/10 = 0.2
- `stayin`: 1/10 = 0.1
- `shopping`: 1/10 = 0.1

Công thức Gini = 1 - Σ(p_i)²

Gini(Root) = 1 - [ (0.6)² + (0.2)² + (0.1)² + (0.1)² ]
Gini(Root) = 1 - [ 0.36 + 0.04 + 0.01 + 0.01 ]
Gini(Root) = 1 - 0.42 = **0.58**

---

## 3. Tính Gini có trọng số cho từng cách chia nhị phân của `Weather`

Thuộc tính `Weather` có 3 giá trị (`sunny`, `windy`, `rainy`), nên có 3 cách chia nhị phân có thể.

### Cách chia 1: `{sunny}` vs. `{windy, rainy}`

#### a. Nhánh Trái: `Weather = sunny` (3 mẫu)
- **Phân bố `Decision`:** `cinema`: 1, `tennis`: 2
- **Tính Gini:**
    Gini(sunny) = 1 - [ (1/3)² + (2/3)² ] = 1 - [ 1/9 + 4/9 ] = 1 - 5/9 ≈ **0.444**

#### b. Nhánh Phải: `Weather` là `{windy, rainy}` (4 + 3 = 7 mẫu)
- **Phân bố `Decision`:** `cinema`: (3+2)=5, `shopping`: (1+0)=1, `stayin`: (0+1)=1
- **Tính Gini:**
    Gini({windy,rainy}) = 1 - [ (5/7)² + (1/7)² + (1/7)² ] = 1 - [ 25/49 + 1/49 + 1/49 ] = 1 - 27/49 ≈ **0.449**

#### c. Gini có trọng số cho cách chia này
- **Tính toán:** (3/10) * Gini(sunny) + (7/10) * Gini({windy,rainy})
            = (0.3 * 0.444) + (0.7 * 0.449)
            = 0.1332 + 0.3143 = **0.4475**

---


### Cách chia 2: `{windy}` vs. `{sunny, rainy}`

#### a. Nhánh Trái: `Weather = windy` (4 mẫu)
- **Phân bố `Decision`:** `cinema`: 3, `shopping`: 1
- **Tính Gini:**
    Gini(windy) = 1 - [ (3/4)² + (1/4)² ] = 1 - [ 9/16 + 1/16 ] = 1 - 10/16 = **0.375**

#### b. Nhánh Phải: `Weather` là `{sunny, rainy}` (3 + 3 = 6 mẫu)
- **Phân bố `Decision`:** `cinema`: (1+2)=3, `tennis`: (2+0)=2, `stayin`: (0+1)=1
- **Tính Gini:**
    Gini({sunny,rainy}) = 1 - [ (3/6)² + (2/6)² + (1/6)² ] = 1 - [ 9/36 + 4/36 + 1/36 ] = 1 - 14/36 ≈ **0.611**

#### c. Gini có trọng số cho cách chia này
- **Tính toán:** (4/10) * Gini(windy) + (6/10) * Gini({sunny,rainy})
            = (0.4 * 0.375) + (0.6 * 0.611)
            = 0.150 + 0.3666 = **0.5166**

---


### Cách chia 3: `{rainy}` vs. `{sunny, windy}`

#### a. Nhánh Trái: `Weather = rainy` (3 mẫu)
- **Phân bố `Decision`:** `cinema`: 2, `stayin`: 1
- **Tính Gini:**
    Gini(rainy) = 1 - [ (2/3)² + (1/3)² ] = 1 - [ 4/9 + 1/9 ] = 1 - 5/9 ≈ **0.444**

#### b. Nhánh Phải: `Weather` là `{sunny, windy}` (3 + 4 = 7 mẫu)
- **Phân bố `Decision`:** `cinema`: (1+3)=4, `tennis`: (2+0)=2, `shopping`: (0+1)=1
- **Tính Gini:**
    Gini({sunny,windy}) = 1 - [ (4/7)² + (2/7)² + (1/7)² ] = 1 - [ 16/49 + 4/49 + 1/49 ] = 1 - 21/49 ≈ **0.571**

#### c. Gini có trọng số cho cách chia này
- **Tính toán:** (3/10) * Gini(rainy) + (7/10) * Gini({sunny,windy})
            = (0.3 * 0.444) + (0.7 * 0.571)
            = 0.1332 + 0.3997 = **0.5329**

---

## 4. Tổng kết và lựa chọn cách chia tốt nhất cho `Weather`

| Cách chia nhị phân (`Weather`) | Gini có trọng số |
|--------------------------------|------------------|
| `{sunny}` vs. `{windy, rainy}` | **0.4475**       |
| `{windy}` vs. `{sunny, rainy}` | 0.5166           |
| `{rainy}` vs. `{sunny, windy}` | 0.5329           |

**Kết luận:** Cách chia **`{sunny}` vs. `{windy, rainy}`** mang lại Gini có trọng số thấp nhất (**0.4475**). Do đó, đây là cách chia tốt nhất cho thuộc tính `Weather` để phân lớp `Decision`.

---

## 5. Cây quyết định đơn giản hóa dựa trên `Weather`

Nếu chỉ sử dụng thuộc tính `Weather` để tạo cây quyết định (tức là coi nó là nút gốc), cây sẽ có cấu trúc như sau:

```
                  [Weather]
                  /       \
             sunny       {windy, rainy}
            /                 \
     [Tennis]               [Node B]
       (1 cinema,            (5 cinema,
        2 tennis)            1 stayin,
                             1 shopping)
```

**Nhánh `sunny`:** Vì có 2 `tennis` và 1 `cinema`, nút này không thuần khiết. Quyết định dự đoán sẽ là `tennis` (do là đa số).

**Nhánh `{windy, rainy}`:** Nút này cũng không thuần khiết, cần phải phân chia tiếp nếu có các thuộc tính khác. Nếu không, quyết định dự đoán sẽ là `cinema` (do là đa số).

```
