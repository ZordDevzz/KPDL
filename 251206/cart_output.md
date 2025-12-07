# Phân tích chi tiết thuật toán CART
Đây là tài liệu hướng dẫn từng bước về cách hoạt động của thuật toán CART (Classification and Regression Trees) cho mục đích giáo dục.

## 1. Tổng quan về Tập dữ liệu (Dataset Overview)
- **Tập dữ liệu:** `data.csv`
- **Tổng số mẫu:** 6
- **Các thuộc tính (Features):** humid, weather, wind
- **Thuộc tính quyết định (Target):** 'decision'

## 2. Tính Chỉ số Gini cho toàn bộ tập dữ liệu
Chỉ số Gini (Gini Impurity) đo lường khả năng một phần tử được chọn ngẫu nhiên sẽ bị phân loại sai. Giá trị càng thấp, tập dữ liệu càng 'tinh khiết'.

### Công thức Gini
`Gini(D) = 1 - Σ(p_i^2)`
- `D`: Tập dữ liệu hiện tại.
- `p_i`: Tỷ lệ của lớp `i` trong tập dữ liệu `D`.

- **Tính toán Gini:**
  - P(no) = 4/6 = 0.667
  - P(no)^2 = 0.444
  - P(yes) = 2/6 = 0.333
  - P(yes)^2 = 0.111
- **Gini** = 1 - (0.444 + 0.111) = **0.444**

=> **Gini ban đầu của tập dữ liệu Gini(D)** = **0.444**

## 3. Tính Gini Index cho từng cách phân chia (Split)
CART tạo ra các cây nhị phân, vì vậy nó sẽ xem xét từng cách chia mỗi thuộc tính thành hai nhóm và tính toán Gini Index (chỉ số Gini có trọng số) cho mỗi cách chia đó.

### Công thức Gini Index
`Gini_index(D, A) = (|D_true|/|D|) * Gini(D_true) + (|D_false|/|D|) * Gini(D_false)`

### 3.1. Phân tích thuộc tính: **'humid'**

#### Phân chia (Split): `humid == high` vs. `humid != high`
- **Nhánh Đúng (True)**: `humid == high` (có 3 mẫu)
  - **Tính toán Gini:**
    - P(no) = 2/3 = 0.667
    - P(no)^2 = 0.444
    - P(yes) = 1/3 = 0.333
    - P(yes)^2 = 0.111
  - **Gini** = 1 - (0.444 + 0.111) = **0.444**
- **Nhánh Sai (False)**: `humid != high` (có 3 mẫu)
  - **Tính toán Gini:**
    - P(no) = 2/3 = 0.667
    - P(no)^2 = 0.444
    - P(yes) = 1/3 = 0.333
    - P(yes)^2 = 0.111
  - **Gini** = 1 - (0.444 + 0.111) = **0.444**
- **Tính Gini Index cho phân chia này:**
  - Gini_index = (Trọng số True * Gini True) + (Trọng số False * Gini False)
  - Gini_index = (0.500 * 0.444) + (0.500 * 0.444) = **0.444**

#### Phân chia (Split): `humid == normal` vs. `humid != normal`
- **Nhánh Đúng (True)**: `humid == normal` (có 3 mẫu)
  - **Tính toán Gini:**
    - P(no) = 2/3 = 0.667
    - P(no)^2 = 0.444
    - P(yes) = 1/3 = 0.333
    - P(yes)^2 = 0.111
  - **Gini** = 1 - (0.444 + 0.111) = **0.444**
- **Nhánh Sai (False)**: `humid != normal` (có 3 mẫu)
  - **Tính toán Gini:**
    - P(no) = 2/3 = 0.667
    - P(no)^2 = 0.444
    - P(yes) = 1/3 = 0.333
    - P(yes)^2 = 0.111
  - **Gini** = 1 - (0.444 + 0.111) = **0.444**
- **Tính Gini Index cho phân chia này:**
  - Gini_index = (Trọng số True * Gini True) + (Trọng số False * Gini False)
  - Gini_index = (0.500 * 0.444) + (0.500 * 0.444) = **0.444**

### 3.2. Phân tích thuộc tính: **'weather'**

#### Phân chia (Split): `weather == overcast` vs. `weather != overcast`
- **Nhánh Đúng (True)**: `weather == overcast` (có 1 mẫu)
  - **Tính toán Gini:**
    - P(yes) = 1/1 = 1.000
    - P(yes)^2 = 1.000
  - **Gini** = 1 - (1.000) = **0.000**
- **Nhánh Sai (False)**: `weather != overcast` (có 5 mẫu)
  - **Tính toán Gini:**
    - P(no) = 4/5 = 0.800
    - P(no)^2 = 0.640
    - P(yes) = 1/5 = 0.200
    - P(yes)^2 = 0.040
  - **Gini** = 1 - (0.640 + 0.040) = **0.320**
- **Tính Gini Index cho phân chia này:**
  - Gini_index = (Trọng số True * Gini True) + (Trọng số False * Gini False)
  - Gini_index = (0.167 * 0.000) + (0.833 * 0.320) = **0.267**

#### Phân chia (Split): `weather == rainy` vs. `weather != rainy`
- **Nhánh Đúng (True)**: `weather == rainy` (có 3 mẫu)
  - **Tính toán Gini:**
    - P(no) = 3/3 = 1.000
    - P(no)^2 = 1.000
  - **Gini** = 1 - (1.000) = **0.000**
- **Nhánh Sai (False)**: `weather != rainy` (có 3 mẫu)
  - **Tính toán Gini:**
    - P(no) = 1/3 = 0.333
    - P(no)^2 = 0.111
    - P(yes) = 2/3 = 0.667
    - P(yes)^2 = 0.444
  - **Gini** = 1 - (0.111 + 0.444) = **0.444**
- **Tính Gini Index cho phân chia này:**
  - Gini_index = (Trọng số True * Gini True) + (Trọng số False * Gini False)
  - Gini_index = (0.500 * 0.000) + (0.500 * 0.444) = **0.222**

#### Phân chia (Split): `weather == sunny` vs. `weather != sunny`
- **Nhánh Đúng (True)**: `weather == sunny` (có 2 mẫu)
  - **Tính toán Gini:**
    - P(no) = 1/2 = 0.500
    - P(no)^2 = 0.250
    - P(yes) = 1/2 = 0.500
    - P(yes)^2 = 0.250
  - **Gini** = 1 - (0.250 + 0.250) = **0.500**
- **Nhánh Sai (False)**: `weather != sunny` (có 4 mẫu)
  - **Tính toán Gini:**
    - P(no) = 3/4 = 0.750
    - P(no)^2 = 0.562
    - P(yes) = 1/4 = 0.250
    - P(yes)^2 = 0.062
  - **Gini** = 1 - (0.562 + 0.062) = **0.375**
- **Tính Gini Index cho phân chia này:**
  - Gini_index = (Trọng số True * Gini True) + (Trọng số False * Gini False)
  - Gini_index = (0.333 * 0.500) + (0.667 * 0.375) = **0.417**

### 3.3. Phân tích thuộc tính: **'wind'**

#### Phân chia (Split): `wind == strong` vs. `wind != strong`
- **Nhánh Đúng (True)**: `wind == strong` (có 2 mẫu)
  - **Tính toán Gini:**
    - P(no) = 2/2 = 1.000
    - P(no)^2 = 1.000
  - **Gini** = 1 - (1.000) = **0.000**
- **Nhánh Sai (False)**: `wind != strong` (có 4 mẫu)
  - **Tính toán Gini:**
    - P(no) = 2/4 = 0.500
    - P(no)^2 = 0.250
    - P(yes) = 2/4 = 0.500
    - P(yes)^2 = 0.250
  - **Gini** = 1 - (0.250 + 0.250) = **0.500**
- **Tính Gini Index cho phân chia này:**
  - Gini_index = (Trọng số True * Gini True) + (Trọng số False * Gini False)
  - Gini_index = (0.333 * 0.000) + (0.667 * 0.500) = **0.333**

#### Phân chia (Split): `wind == weak` vs. `wind != weak`
- **Nhánh Đúng (True)**: `wind == weak` (có 4 mẫu)
  - **Tính toán Gini:**
    - P(no) = 2/4 = 0.500
    - P(no)^2 = 0.250
    - P(yes) = 2/4 = 0.500
    - P(yes)^2 = 0.250
  - **Gini** = 1 - (0.250 + 0.250) = **0.500**
- **Nhánh Sai (False)**: `wind != weak` (có 2 mẫu)
  - **Tính toán Gini:**
    - P(no) = 2/2 = 1.000
    - P(no)^2 = 1.000
  - **Gini** = 1 - (1.000) = **0.000**
- **Tính Gini Index cho phân chia này:**
  - Gini_index = (Trọng số True * Gini True) + (Trọng số False * Gini False)
  - Gini_index = (0.667 * 0.500) + (0.333 * 0.000) = **0.333**

## 4. Tổng kết và Chọn cách phân chia tốt nhất
So sánh Gini Index thấp nhất từ mỗi thuộc tính.

| Thuộc tính (Feature) | Cách chia tốt nhất (Best Split) | Gini Index thấp nhất |
|---|---|---|
| weather | 'rainy' vs Others | 0.222 |
| wind | 'strong' vs Others | 0.333 |
| humid | 'high' vs Others | 0.444 |

**=> Kết luận:** Cách phân chia có Gini Index thấp nhất tổng thể là của thuộc tính **'weather'** khi chia theo điều kiện **`weather == rainy`** (Gini Index = **0.222**). Đây sẽ là nút gốc của cây.

## 5. Xây dựng Cây quyết định (Decision Tree)
Cây được xây dựng bằng cách lặp lại quy trình trên cho mỗi nút con.

```
Nếu (If) **weather == rainy**:
├── **Đúng (True):**
│   └── **Quyết định (Decision): no**
└── **Sai (False):**
    Nếu (If) **humid == high**:
    ├── **Đúng (True):**
    │   └── **Quyết định (Decision): no**
    └── **Sai (False):**
        └── **Quyết định (Decision): yes**

```