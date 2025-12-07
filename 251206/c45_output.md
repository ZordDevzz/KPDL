# Phân tích chi tiết thuật toán C4.5
Đây là tài liệu hướng dẫn từng bước về cách hoạt động của thuật toán C4.5 cho mục đích giáo dục.

## 1. Tổng quan về Tập dữ liệu (Dataset Overview)
- **Tập dữ liệu:** `data.csv`
- **Tổng số mẫu:** 6
- **Các thuộc tính (Features):** humid, weather, wind
- **Thuộc tính quyết định (Target):** 'decision'

## 2. Tính Entropy cho toàn bộ tập dữ liệu
Entropy (độ hỗn loạn) đo lường sự không chắc chắn trong tập dữ liệu. Giá trị càng cao, dữ liệu càng hỗn loạn.

### Công thức Entropy (I)
`I = Entropy(S) = - Σ (p_i * log2(p_i))`
- `S`: Tập dữ liệu hiện tại.
- `p_i`: Tỷ lệ của lớp `i` trong tập dữ liệu `S`.

- **Tính toán Entropy:**
  - P(no) = 4/6 = 0.667
  - Term: -(0.667 * log2(0.667)) = 0.390
  - P(yes) = 2/6 = 0.333
  - Term: -(0.333 * log2(0.333)) = 0.528
- **Entropy** = 0.390 + 0.528 = **0.918**

=> **Entropy ban đầu của tập dữ liệu I(S)** = **0.918**

## 3. Tính Gain Ratio cho từng thuộc tính
C4.5 sử dụng Gain Ratio để chọn thuộc tính tốt nhất để phân chia cây. Nó giải quyết vấn đề của Information Gain khi có các thuộc tính với nhiều giá trị khác nhau.

### 3.1. Phân tích thuộc tính: **'humid'**

#### a) Tính Lợi ích Thông tin (Information Gain)
Công thức: `Gain(S, A) = Entropy(S) - Σ ((|S_v| / |S|) * Entropy(S_v))`

- **Tính Entropy cho từng giá trị con (Entropy(S_v)):**
  - **Với giá trị 'high'** (có 3 mẫu):
    - **Tính toán Entropy:**
      - P(no) = 2/3 = 0.667
      - Term: -(0.667 * log2(0.667)) = 0.390
      - P(yes) = 1/3 = 0.333
      - Term: -(0.333 * log2(0.333)) = 0.528
    - **Entropy** = 0.390 + 0.528 = **0.918**
    - Trọng số (Weight) = 3/6 = 0.500
    - Entropy có trọng số = 0.500 * 0.918 = 0.459

  - **Với giá trị 'normal'** (có 3 mẫu):
    - **Tính toán Entropy:**
      - P(no) = 2/3 = 0.667
      - Term: -(0.667 * log2(0.667)) = 0.390
      - P(yes) = 1/3 = 0.333
      - Term: -(0.333 * log2(0.333)) = 0.528
    - **Entropy** = 0.390 + 0.528 = **0.918**
    - Trọng số (Weight) = 3/6 = 0.500
    - Entropy có trọng số = 0.500 * 0.918 = 0.459

- **Tổng Entropy có trọng số** = **0.918**
- **Gain('humid')** = 0.918 - 0.918 = **0.000**

#### b) Tính Thông tin Phân tách (Split Information - SI)
Công thức: `SI(A) = - Σ ((|S_v| / |S|) * log2(|S_v| / |S|))`

- **Với giá trị 'high'**: -(3/6) * log2(3/6) = 0.500
- **Với giá trị 'normal'**: -(3/6) * log2(3/6) = 0.500
- **SI('humid')** = 0.500 + 0.500 = **1.000**

#### c) Tính Tỷ lệ Lợi ích (Gain Ratio - GR)
Công thức: `GR(A) = Gain(A) / SI(A)`
- **GR('humid')** = 0.000 / 1.000 = **0.000**

### 3.2. Phân tích thuộc tính: **'weather'**

#### a) Tính Lợi ích Thông tin (Information Gain)
Công thức: `Gain(S, A) = Entropy(S) - Σ ((|S_v| / |S|) * Entropy(S_v))`

- **Tính Entropy cho từng giá trị con (Entropy(S_v)):**
  - **Với giá trị 'overcast'** (có 1 mẫu):
    - **Tính toán Entropy:**
      - P(yes) = 1/1 = 1.000
      - Term: -(1.000 * log2(1.000)) = -0.000
    - **Entropy** = -0.000 = **0.000**
    - Trọng số (Weight) = 1/6 = 0.167
    - Entropy có trọng số = 0.167 * 0.000 = 0.000

  - **Với giá trị 'rainy'** (có 3 mẫu):
    - **Tính toán Entropy:**
      - P(no) = 3/3 = 1.000
      - Term: -(1.000 * log2(1.000)) = -0.000
    - **Entropy** = -0.000 = **0.000**
    - Trọng số (Weight) = 3/6 = 0.500
    - Entropy có trọng số = 0.500 * 0.000 = 0.000

  - **Với giá trị 'sunny'** (có 2 mẫu):
    - **Tính toán Entropy:**
      - P(no) = 1/2 = 0.500
      - Term: -(0.500 * log2(0.500)) = 0.500
      - P(yes) = 1/2 = 0.500
      - Term: -(0.500 * log2(0.500)) = 0.500
    - **Entropy** = 0.500 + 0.500 = **1.000**
    - Trọng số (Weight) = 2/6 = 0.333
    - Entropy có trọng số = 0.333 * 1.000 = 0.333

- **Tổng Entropy có trọng số** = **0.333**
- **Gain('weather')** = 0.918 - 0.333 = **0.585**

#### b) Tính Thông tin Phân tách (Split Information - SI)
Công thức: `SI(A) = - Σ ((|S_v| / |S|) * log2(|S_v| / |S|))`

- **Với giá trị 'overcast'**: -(1/6) * log2(1/6) = 0.431
- **Với giá trị 'rainy'**: -(3/6) * log2(3/6) = 0.500
- **Với giá trị 'sunny'**: -(2/6) * log2(2/6) = 0.528
- **SI('weather')** = 0.431 + 0.500 + 0.528 = **1.459**

#### c) Tính Tỷ lệ Lợi ích (Gain Ratio - GR)
Công thức: `GR(A) = Gain(A) / SI(A)`
- **GR('weather')** = 0.585 / 1.459 = **0.401**

### 3.3. Phân tích thuộc tính: **'wind'**

#### a) Tính Lợi ích Thông tin (Information Gain)
Công thức: `Gain(S, A) = Entropy(S) - Σ ((|S_v| / |S|) * Entropy(S_v))`

- **Tính Entropy cho từng giá trị con (Entropy(S_v)):**
  - **Với giá trị 'strong'** (có 2 mẫu):
    - **Tính toán Entropy:**
      - P(no) = 2/2 = 1.000
      - Term: -(1.000 * log2(1.000)) = -0.000
    - **Entropy** = -0.000 = **0.000**
    - Trọng số (Weight) = 2/6 = 0.333
    - Entropy có trọng số = 0.333 * 0.000 = 0.000

  - **Với giá trị 'weak'** (có 4 mẫu):
    - **Tính toán Entropy:**
      - P(no) = 2/4 = 0.500
      - Term: -(0.500 * log2(0.500)) = 0.500
      - P(yes) = 2/4 = 0.500
      - Term: -(0.500 * log2(0.500)) = 0.500
    - **Entropy** = 0.500 + 0.500 = **1.000**
    - Trọng số (Weight) = 4/6 = 0.667
    - Entropy có trọng số = 0.667 * 1.000 = 0.667

- **Tổng Entropy có trọng số** = **0.667**
- **Gain('wind')** = 0.918 - 0.667 = **0.252**

#### b) Tính Thông tin Phân tách (Split Information - SI)
Công thức: `SI(A) = - Σ ((|S_v| / |S|) * log2(|S_v| / |S|))`

- **Với giá trị 'strong'**: -(2/6) * log2(2/6) = 0.528
- **Với giá trị 'weak'**: -(4/6) * log2(4/6) = 0.390
- **SI('wind')** = 0.528 + 0.390 = **0.918**

#### c) Tính Tỷ lệ Lợi ích (Gain Ratio - GR)
Công thức: `GR(A) = Gain(A) / SI(A)`
- **GR('wind')** = 0.252 / 0.918 = **0.274**

## 4. Tổng kết và Chọn thuộc tính tốt nhất
| Thuộc tính (Feature) | Gain (G) | Split Info (SI) | Gain Ratio (GR) |
|---|---|---|---|
| weather | 0.585 | 1.459 | 0.401 |
| wind | 0.252 | 0.918 | 0.274 |
| humid | 0.000 | 1.000 | 0.000 |

**=> Kết luận:** Thuộc tính có Gain Ratio cao nhất là **'weather'** (GR = 0.401). Do đó, nó sẽ được chọn làm nút gốc của cây quyết định.

## 5. Xây dựng Cây quyết định (Decision Tree)
Nút gốc (Root): **weather**

```
├── Nếu (If) **weather = overcast**:
│   └── **Quyết định (Decision): yes**
├── Nếu (If) **weather = rainy**:
│   └── **Quyết định (Decision): no**
└── Nếu (If) **weather = sunny**:
    ├── Nếu (If) **humid = high**:
    │   └── **Quyết định (Decision): no**
    └── Nếu (If) **humid = normal**:
        └── **Quyết định (Decision): yes**

```