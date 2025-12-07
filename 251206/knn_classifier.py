import csv
from collections import Counter
import math

def one_hot_encode(data, feature_names, decision_index):
    value_map = {name: sorted(list(set(row[i] for row in data))) for i, name in enumerate(feature_names)}
    encoded_data = []
    original_data_with_encoded = []
    for row in data:
        encoded_row = []
        for i, feature_name in enumerate(feature_names):
            binary_vector = [1 if value == row[i] else 0 for value in value_map[feature_name]]
            encoded_row.extend(binary_vector)
        encoded_data.append((encoded_row, row[decision_index]))
        original_data_with_encoded.append(row + [encoded_row])
    return encoded_data, value_map, original_data_with_encoded

def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1)):
        distance += (row1[i] - row2[i])**2
    return math.sqrt(distance)

def run_knn_detailed_example(csv_file, new_instance, k):
    output = []
    output.append("# Phân tích chi tiết thuật toán k-Nearest Neighbors (k-NN)")
    output.append("Đây là tài liệu hướng dẫn từng bước về cách hoạt động của thuật toán k-NN cho mục đích giáo dục.\n")
    output.append("k-NN là một thuật toán 'học lười' (lazy learning), nghĩa là nó không xây dựng một mô hình rõ ràng. Thay vào đó, nó lưu trữ toàn bộ tập dữ liệu huấn luyện và thực hiện dự đoán dựa trên sự tương đồng (khoảng cách) với các điểm dữ liệu mới.\n")

    # Load data
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        raw_data = list(reader)

    clean_header = [h for h in header if h]
    feature_names = clean_header[1:-1]
    
    data = []
    original_rows = []
    for row in raw_data:
        clean_row = [val for val in row if val]
        if len(clean_row) == len(clean_header):
            original_rows.append(clean_row)
            data.append(clean_row[1:])

    decision_index = len(feature_names)
    
    output.append(f"## 1. Dữ liệu và Tham số")
    output.append(f"- **Tập dữ liệu:** `{csv_file}`")
    output.append(f"- **Giá trị k:** {k}")
    output.append(f"- **Mẫu mới cần phân loại:** `{new_instance}`\n")
    
    output.append("### Dữ liệu gốc:")
    output.append(f"| {' | '.join(clean_header)} |")
    output.append(f"|{'---|' * len(clean_header)}")
    for row in original_rows:
        output.append(f"| {' | '.join(row)} |")

    # Step 2: One-Hot Encoding
    encoded_data, value_map, original_data_with_encoded = one_hot_encode(data, feature_names, decision_index)
    output.append("\n## 2. Mã hóa One-Hot (One-Hot Encoding)")
    output.append("Vì các thuộc tính của chúng ta là dạng chữ (categorical), chúng ta cần chuyển chúng thành dạng số để có thể tính toán khoảng cách. Mã hóa One-Hot là một phương pháp phổ biến để làm việc này.\n")
    output.append("### Sơ đồ mã hóa:")
    encoded_instance = []
    for feature_name in feature_names:
        output.append(f"- **{feature_name}**:")
        binary_vector = []
        for val in value_map[feature_name]:
            is_present = 1 if val == new_instance[feature_name] else 0
            binary_vector.append(is_present)
            output.append(f"  - `{val}` -> `{''.join(str(v) for v in ([1 if v == val else 0 for v in value_map[feature_name]]))}`")
        encoded_instance.extend(binary_vector)

    output.append("\n### Dữ liệu sau khi mã hóa:")
    output.append(f"| {' | '.join(clean_header)} | Vector mã hóa |")
    output.append(f"|{'---|' * (len(clean_header) + 1)}")
    for row in original_data_with_encoded:
        output.append(f"| {' | '.join(row[:-1])} | `{row[-1]}` |")

    output.append(f"\n- **Vector mã hóa cho mẫu mới:** `{encoded_instance}`\n")

    # Step 3: Distance Calculation
    output.append("## 3. Tính khoảng cách Euclid (Euclidean Distance)")
    output.append("Bây giờ, chúng ta tính khoảng cách từ mẫu mới đến TẤT CẢ các mẫu trong tập dữ liệu huấn luyện.\n")
    output.append("### Công thức:")
    output.append("`Distance(A, B) = sqrt( Σ(A_i - B_i)^2 )`\n")
    
    distances = []
    output.append("| ID Gốc | Vector Huấn luyện | Vector Mới | Tính toán Khoảng cách | Kết quả (Distance) | Quyết định |")
    output.append("|---|---|---|---|---|---|")
    for i, (train_vec, decision) in enumerate(encoded_data):
        dist = euclidean_distance(encoded_instance, train_vec)
        distances.append((original_rows[i][0], decision, dist))
        calc_str = " + ".join([f"({v1}-{v2})^2" for v1, v2 in zip(train_vec, encoded_instance)])
        output.append(f"| {original_rows[i][0]} | `{train_vec}` | `{encoded_instance}` | `sqrt({calc_str})` | **{dist:.3f}** | {decision} |")

    # Step 4: Find Neighbors
    distances.sort(key=lambda tup: tup[2])
    output.append(f"\n## 4. Sắp xếp và tìm {k} láng giềng gần nhất")
    output.append("Sắp xếp tất cả các điểm dữ liệu theo khoảng cách tăng dần và chọn ra k điểm đầu tiên.\n")
    output.append("| ID Gốc | Quyết định | Khoảng cách | Là láng giềng? |")
    output.append("|---|---|---|---|")
    neighbors = []
    for i in range(len(distances)):
        is_neighbor = "✅" if i < k else "❌"
        if i < k:
            neighbors.append(distances[i])
        output.append(f"| {distances[i][0]} | {distances[i][1]} | {distances[i][2]:.3f} | {is_neighbor} |")

    # Step 5: Prediction
    output.append("\n## 5. Dự đoán dựa trên bỏ phiếu đa số")
    output.append(f"Các láng giềng được chọn là: {', '.join([f'{n[0]} ({n[1]})' for n in neighbors])}.\n")
    neighbor_decisions = [row[1] for row in neighbors]
    prediction = Counter(neighbor_decisions).most_common(1)[0][0]
    
    output.append("- **Các phiếu bầu (votes) từ các láng giềng:**")
    votes = Counter(neighbor_decisions)
    for dec, count in votes.items():
        output.append(f"  - Lớp '{dec}': {count} phiếu")
        
    output.append(f"\n**=> Kết luận:** Lớp có nhiều phiếu bầu nhất là **'{prediction}'**. Do đó, đây là dự đoán cuối cùng cho mẫu mới.")
    
    return "\n".join(output)

if __name__ == "__main__":
    new_data_point = {
        'humid': 'normal',
        'weather': 'sunny',
        'wind': 'weak'
    }
    K = 3
    
    output_content = run_knn_detailed_example('data.csv', new_data_point, K)
    
    with open('knn_output.md', 'w', encoding='utf-8') as f:
        f.write(output_content)
    print("Đã tạo tệp knn_output.md với phân tích chi tiết.")
