# Phân tích dự đoán bằng thuật toán Naive Bayes

Tài liệu này giải thích chi tiết các bước của thuật toán Naive Bayes để dự đoán cột "Job Offer" cho dữ liệu trong tệp `predict.csv` dựa trên tập dữ liệu `dataset.csv`.

## 1. Tổng quan về Thuật toán Naive Bayes

Naive Bayes là một thuật toán phân loại dựa trên **Định lý Bayes**. Thuật toán này được gọi là "ngây thơ" (naive) vì nó đưa ra một giả định đơn giản hóa: tất cả các đặc trưng (features) là **độc lập với nhau** khi biết lớp (class).

Công thức của Định lý Bayes:

**P(Y | X) = [P(X | Y) * P(Y)] / P(X)**

Trong đó:
- **P(Y | X)**: Xác suất có điều kiện của lớp Y khi biết đặc trưng X (posterior). Đây là xác suất chúng ta muốn tính.
- **P(X | Y)**: Xác suất có điều kiện của đặc trưng X khi biết lớp Y (likelihood).
- **P(Y)**: Xác suất của lớp Y (prior).
- **P(X)**: Xác suất của đặc trưng X (evidence).

Để dự đoán, chúng ta tính P(Y | X) cho tất cả các lớp Y và chọn lớp có xác suất cao nhất.

## 2. Dữ liệu

### Tập dữ liệu huấn luyện (`dataset.csv`)

| ID  | Practical Knowledge | Comm Skill | CGPA | Interactive | Job Offer |
|:----|:--------------------|:-----------|:-----|:------------|:----------|
| A1  | Very good           | Good       | A    | TRUE        | Yes       |
| A2  | Good                | Morderate  | B    | FALSE       | Yes       |
| A3  | Average             | Poor       | A    | FALSE       | No        |
| A4  | Average             | Good       | C    | FALSE       | No        |
| A5  | Good                | Morderate  | B    | TRUE        | Yes       |
| A6  | Good                | Morderate  | A    | TRUE        | Yes       |
| A7  | Good                | Poor       | C    | TRUE        | No        |
| A8  | Very good           | Good       | A    | FALSE       | Yes       |
| A9  | Good                | Good       | B    | TRUE        | Yes       |
| A10 | Average             | Good       | B    | TRUE        | Yes       |

### Dữ liệu cần dự đoán (`predict.csv`)

Dữ liệu đầu vào **X** để dự đoán là:
- **Practical Knowledge**: Good
- **Comm Skill**: Good
- **CGPA**: A
- **Interactive**: FALSE

## 3. Các bước tính toán

### Bước 1: Tính xác suất tiên nghiệm (Prior Probabilities) - P(Y)

Chúng ta tính xác suất cho mỗi lớp "Job Offer" trong tập dữ liệu.

- Tổng số mẫu: 10
- Số lượng `Job Offer = Yes`: 7
- Số lượng `Job Offer = No`: 3

Xác suất tiên nghiệm là:
- **P(Job Offer = Yes) = 7/10 = 0.7**
- **P(Job Offer = No) = 3/10 = 0.3**

### Bước 2: Tính xác suất có điều kiện (Likelihoods) - P(X | Y)

Chúng ta tính xác suất của mỗi đặc trưng (feature) với điều kiện là một lớp cụ thể.

#### Bảng tần suất cho `Job Offer = Yes` (7 mẫu)

| Đặc trưng            | Giá trị | Tần suất | Xác suất P(feature \| Yes) |
|:----------------------|:--------|:---------|:--------------------------|
| Practical Knowledge   | Good    | 4        | 4/7                       |
| Comm Skill            | Good    | 3        | 3/7                       |
| CGPA                  | A       | 3        | 3/7                       |
| Interactive           | FALSE   | 2        | 2/7                       |

#### Bảng tần suất cho `Job Offer = No` (3 mẫu)

| Đặc trưng            | Giá trị | Tần suất | Xác suất P(feature \| No)  |
|:----------------------|:--------|:---------|:-------------------------|
| Practical Knowledge   | Good    | 0        | **0/3 = 0**              |
| Comm Skill            | Good    | 1        | 1/3                      |
| CGPA                  | A       | 1        | 1/3                      |
| Interactive           | FALSE   | 2        | 2/3                      |

**Lưu ý quan trọng (Vấn đề tần suất bằng 0):**
Xác suất `P(Practical Knowledge = Good | No)` bằng 0. Khi nhân các xác suất lại với nhau, điều này sẽ làm cho toàn bộ xác suất của lớp `No` trở thành 0, có thể dẫn đến kết quả sai. Để giải quyết vấn đề này, chúng ta sử dụng một kỹ thuật gọi là **Làm mịn Laplace (Laplace Smoothing)**.

#### Bước 2 (Nâng cao): Tính Likelihoods với Làm mịn Laplace

Công thức làm mịn Laplace:
**P(feature | class) = (k + 1) / (N + V)**
- `k`: Số lần xuất hiện của giá trị đặc trưng trong lớp đó.
- `N`: Tổng số mẫu của lớp đó.
- `V`: Tổng số giá trị duy nhất của đặc trưng đó.

- **V (Practical Knowledge)** = 3 (Very good, Good, Average)
- **V (Comm Skill)** = 3 (Good, Morderate, Poor)
- **V (CGPA)** = 3 (A, B, C)
- **V (Interactive)** = 2 (TRUE, FALSE)

**Likelihoods đã làm mịn cho `Job Offer = Yes` (N=7):**
- P(Practical Knowledge = Good | Yes) = (4 + 1) / (7 + 3) = 5/10 = 0.5
- P(Comm Skill = Good | Yes) = (3 + 1) / (7 + 3) = 4/10 = 0.4
- P(CGPA = A | Yes) = (3 + 1) / (7 + 3) = 4/10 = 0.4
- P(Interactive = FALSE | Yes) = (2 + 1) / (7 + 2) = 3/9 ≈ 0.333

**Likelihoods đã làm mịn cho `Job Offer = No` (N=3):**
- P(Practical Knowledge = Good | No) = (0 + 1) / (3 + 3) = 1/6 ≈ 0.167
- P(Comm Skill = Good | No) = (1 + 1) / (3 + 3) = 2/6 ≈ 0.333
- P(CGPA = A | No) = (1 + 1) / (3 + 3) = 2/6 ≈ 0.333
- P(Interactive = FALSE | No) = (2 + 1) / (3 + 2) = 3/5 = 0.6

### Bước 3: Tính xác suất hậu nghiệm (Posterior Probabilities) - P(Y | X)

Bây giờ chúng ta nhân các xác suất đã tính để tìm ra xác suất cho mỗi lớp.

#### Đối với lớp `Job Offer = Yes`:

P(Yes | X) ∝ P(Yes) * P(Practical Knowledge=Good | Yes) * P(Comm Skill=Good | Yes) * P(CGPA=A | Yes) * P(Interactive=FALSE | Yes)
P(Yes | X) ∝ 0.7 * 0.5 * 0.4 * 0.4 * 0.333
**P(Yes | X) ∝ 0.018648**

#### Đối với lớp `Job Offer = No`:

P(No | X) ∝ P(No) * P(Practical Knowledge=Good | No) * P(Comm Skill=Good | No) * P(CGPA=A | No) * P(Interactive=FALSE | No)
P(No | X) ∝ 0.3 * 0.167 * 0.333 * 0.333 * 0.6
**P(No | X) ∝ 0.00999**

### Bước 4: Đưa ra dự đoán

So sánh hai xác suất hậu nghiệm:
- **P(Yes | X) ∝ 0.018648**
- **P(No | X) ∝ 0.00999**

Vì `0.018648 > 0.00999`, xác suất lớp `Yes` cao hơn.

## 4. Kết luận

Dựa trên thuật toán Naive Bayes và dữ liệu huấn luyện, dự đoán cho mẫu `A11` là:
**Job Offer = Yes**
