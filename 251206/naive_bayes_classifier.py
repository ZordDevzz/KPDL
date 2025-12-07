import csv
from collections import defaultdict, Counter

def run_naive_bayes_detailed_example(csv_file, new_instance):
    output = []
    output.append("# Phân tích chi tiết thuật toán Naive Bayes")
    output.append("Đây là tài liệu hướng dẫn từng bước về cách hoạt động của thuật toán Naive Bayes cho mục đích giáo dục.\n")
    output.append("Naive Bayes là một thuật toán phân loại dựa trên Định lý Bayes với giả định 'ngây thơ' (naive) rằng các thuộc tính là độc lập với nhau khi biết lớp quyết định. Mặc dù giả định này hiếm khi đúng trong thực tế, thuật toán vẫn hoạt động hiệu quả trong nhiều trường hợp.\n")

    # Load data
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        raw_data = list(reader)

    clean_header = [h for h in header if h]
    feature_names = clean_header[1:-1]
    target_name = clean_header[-1]

    data = []
    for row in raw_data:
        clean_row = [val for val in row if val]
        if len(clean_row) == len(clean_header):
            data.append(clean_row[1:])

    total_data_points = len(data)
    
    output.append(f"## 1. Dữ liệu và Phân tích Tần suất")
    output.append(f"- **Tập dữ liệu:** `{csv_file}`")
    output.append(f"- **Mẫu mới cần phân loại:** `{new_instance}`\n")
    
    # --- Training Phase ---
    output.append("## 2. Giai đoạn 'Huấn luyện': Tính các xác suất")
    output.append("Trong Naive Bayes, 'huấn luyện' đơn giản là tính toán các xác suất từ tập dữ liệu đã cho.\n")

    # a) Prior Probabilities
    output.append("### a) Tính Xác suất Tiên nghiệm P(Class) - (Prior Probabilities)")
    output.append("Đây là xác suất xuất hiện của mỗi lớp trong toàn bộ tập dữ liệu.\n")
    class_counts = Counter(row[-1] for row in data)
    class_probabilities = {cls: count / total_data_points for cls, count in class_counts.items()}
    for cls, count in class_counts.items():
        output.append(f"- **P(decision={cls})** = (Số lần '{cls}' xuất hiện) / (Tổng số mẫu) = {count}/{total_data_points} = **{class_probabilities[cls]:.3f}**")

    # b) Conditional Probabilities
    output.append("\n### b) Tính Xác suất có điều kiện P(Thuộc tính | Class) - (Conditional Probabilities)")
    output.append("Đây là xác suất của một giá trị thuộc tính cụ thể, biết trước lớp của nó. Chúng ta sử dụng kỹ thuật **Làm mịn Laplace (add-1)** để tránh xác suất bằng 0 khi một giá trị không xuất hiện trong một lớp nào đó.\n")
    output.append("Công thức (với làm mịn Laplace): `P(Value | Class) = (Số lần Value xuất hiện trong Class + 1) / (Tổng số mẫu của Class + Số lượng giá trị khác nhau của thuộc tính)`\n")
    
    conditional_probabilities = defaultdict(lambda: defaultdict(dict))
    for i, feature_name in enumerate(feature_names):
        output.append(f"#### Phân tích thuộc tính: `{feature_name}`")
        unique_feature_values = sorted(list(set(row[i] for row in data)))
        num_unique_values = len(unique_feature_values)
        
        for cls in sorted(class_counts.keys()):
            class_subset = [row for row in data if row[-1] == cls]
            class_subset_count = len(class_subset)
            feature_values_in_class = [row[i] for row in class_subset]
            feature_counts_in_class = Counter(feature_values_in_class)

            for value in unique_feature_values:
                count = feature_counts_in_class.get(value, 0)
                prob = (count + 1) / (class_subset_count + num_unique_values)
                conditional_probabilities[feature_name][value][cls] = prob
                output.append(f"- **P({feature_name}={value} | decision={cls})** = ({count} + 1) / ({class_subset_count} + {num_unique_values}) = **{prob:.3f}**")
    
    # --- Prediction Phase ---
    output.append("\n## 3. Giai đoạn Dự đoán")
    output.append(f"Áp dụng các xác suất đã tính để dự đoán lớp cho mẫu mới: `{new_instance}`\n")
    output.append("### Công thức dự đoán của Naive Bayes:")
    output.append("Tìm lớp `C` để tối đa hóa biểu thức sau:")
    output.append("`P(C | Features) ∝ P(C) * P(Feature_1 | C) * P(Feature_2 | C) * ...`\n")

    posteriors = {}
    
    output.append("### Tính toán cho từng lớp:")
    for cls, class_prob in class_probabilities.items():
        output.append(f"#### Lớp (Class): **{cls}**")
        posterior = class_prob
        calc_str = f"{class_prob:.3f} (P(decision={cls}))"
        
        for feature_name in feature_names:
            value = new_instance[feature_name]
            cond_prob = conditional_probabilities[feature_name][value][cls]
            posterior *= cond_prob
            calc_str += f" * {cond_prob:.3f} (P({feature_name}={value}|{cls}))"
            
        posteriors[cls] = posterior
        output.append(f"- **Tính toán:** {calc_str}")
        output.append(f"- **Kết quả (chưa chuẩn hóa):** {posterior:.5f}\n")

    output.append("## 4. Tổng kết và Đưa ra Quyết định")
    output.append("So sánh các kết quả tính toán để tìm ra lớp có xác suất cao nhất.\n")
    output.append("| Lớp (Class) | Điểm xác suất (Posterior Score) |")
    output.append("|---|---|")
    for cls, score in posteriors.items():
        output.append(f"| {cls} | {score:.5f} |")

    best_class = max(posteriors, key=posteriors.get)
    output.append(f"\n**=> Kết luận:** Lớp có điểm xác suất cao nhất là **'{best_class}'**. Đây là dự đoán cuối cùng.\n")
    
    # Normalization (optional but good for explanation)
    total_posterior = sum(posteriors.values())
    if total_posterior > 0:
        normalized_posteriors = {cls: post / total_posterior for cls, post in posteriors.items()}
        output.append("### Phụ lục: Chuẩn hóa Xác suất (Optional)")
        output.append("Để chuyển các điểm thành xác suất có tổng bằng 1, chúng ta có thể chuẩn hóa chúng:\n")
        output.append("| Lớp (Class) | Xác suất đã chuẩn hóa P(Class | Features) |")
        output.append("|---|---|")
        for cls, prob in normalized_posteriors.items():
            output.append(f"| {cls} | {prob:.3f} |")

    return "\n".join(output)

if __name__ == "__main__":
    new_data_point = {
        'humid': 'normal',
        'weather': 'sunny',
        'wind': 'weak'
    }
    
    output_content = run_naive_bayes_detailed_example('data.csv', new_data_point)
    
    with open('naive_bayes_output.md', 'w', encoding='utf-8') as f:
        f.write(output_content)
    print("Đã tạo tệp naive_bayes_output.md với phân tích chi tiết.")
