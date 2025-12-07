import csv
from collections import Counter

def calculate_gini(data_column, output_lines=None, indent=""):
    """
    Calculates the Gini impurity of a given data column, optionally printing steps.
    """
    if not data_column:
        return 0.0

    counts = Counter(data_column)
    total_elements = len(data_column)
    gini = 1.0
    
    if output_lines is not None:
        output_lines.append(f"{indent}- **Tính toán Gini:**")
        
    sum_sq_prob = 0
    for cls, count in counts.items():
        probability = count / total_elements
        sum_sq_prob += probability**2
        gini -= probability**2
        if output_lines is not None:
            output_lines.append(f"{indent}  - P({cls}) = {count}/{total_elements} = {probability:.3f}")
            output_lines.append(f"{indent}  - P({cls})^2 = {probability**2:.3f}")

    if output_lines is not None:
        sum_terms = " + ".join([f"{(c/total_elements)**2:.3f}" for c in counts.values()])
        output_lines.append(f"{indent}- **Gini** = 1 - ({sum_terms}) = **{gini:.3f}**")
        
    return gini

def find_best_cart_split(data, feature_names):
    best_gini = 1; best_feature_name = None; best_split_value = None
    target_index = len(data[0]) - 1
    for i, feature_name in enumerate(feature_names):
        feature_index = i
        unique_values = set(row[feature_index] for row in data)
        for value in unique_values:
            subset1 = [row for row in data if row[feature_index] == value]
            subset2 = [row for row in data if row[feature_index] != value]
            if not subset1 or not subset2: continue
            gini1 = calculate_gini([row[target_index] for row in subset1])
            gini2 = calculate_gini([row[target_index] for row in subset2])
            weight1 = len(subset1) / len(data); weight2 = len(subset2) / len(data)
            current_gini = weight1 * gini1 + weight2 * gini2
            if current_gini < best_gini:
                best_gini = current_gini; best_feature_name = feature_name; best_split_value = value
    return best_feature_name, best_split_value

def build_cart_tree(data, feature_names):
    target_index = len(data[0]) - 1
    target_column = [row[target_index] for row in data]
    if len(set(target_column)) == 1: return target_column[0]
    if not feature_names: return Counter(target_column).most_common(1)[0][0]
    best_feature, best_value = find_best_cart_split(data, feature_names)
    if best_feature is None: return Counter(target_column).most_common(1)[0][0]
    tree = {f"{best_feature} == {best_value}": {}}
    feature_index = feature_names.index(best_feature)
    subset_true = [row for row in data if row[feature_index] == best_value]
    subset_false = [row for row in data if row[feature_index] != best_value]
    remaining_features = [f for f in feature_names if f != best_feature] # Typically CART can reuse features, but for simplicity here we remove it
    tree[f"{best_feature} == {best_value}"]["True"] = build_cart_tree(subset_true, remaining_features)
    tree[f"{best_feature} == {best_value}"]["False"] = build_cart_tree(subset_false, remaining_features)
    return tree

def format_cart_tree(tree, indent=""):
    if not isinstance(tree, dict):
        return f"{indent}└── **Quyết định (Decision): {tree}**\n"
    lines = []
    condition = list(tree.keys())[0]
    lines.append(f"{indent}Nếu (If) **{condition}**:\n")
    lines.append(f"{indent}├── **Đúng (True):**\n")
    lines.append(format_cart_tree(tree[condition]["True"], indent + "│   "))
    lines.append(f"{indent}└── **Sai (False):**\n")
    lines.append(format_cart_tree(tree[condition]["False"], indent + "    "))
    return "".join(lines)

def process_data_for_cart_detailed(csv_file):
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

    target_index = len(data[0]) - 1
    output = []
    output.append("# Phân tích chi tiết thuật toán CART")
    output.append("Đây là tài liệu hướng dẫn từng bước về cách hoạt động của thuật toán CART (Classification and Regression Trees) cho mục đích giáo dục.\n")
    output.append(f"## 1. Tổng quan về Tập dữ liệu (Dataset Overview)")
    output.append(f"- **Tập dữ liệu:** `{csv_file}`")
    output.append(f"- **Tổng số mẫu:** {len(data)}")
    output.append(f"- **Các thuộc tính (Features):** {', '.join(feature_names)}")
    output.append(f"- **Thuộc tính quyết định (Target):** '{target_name}'\n")
    
    output.append("## 2. Tính Chỉ số Gini cho toàn bộ tập dữ liệu")
    output.append("Chỉ số Gini (Gini Impurity) đo lường khả năng một phần tử được chọn ngẫu nhiên sẽ bị phân loại sai. Giá trị càng thấp, tập dữ liệu càng 'tinh khiết'.\n")
    output.append("### Công thức Gini")
    output.append("`Gini(D) = 1 - Σ(p_i^2)`")
    output.append("- `D`: Tập dữ liệu hiện tại.")
    output.append("- `p_i`: Tỷ lệ của lớp `i` trong tập dữ liệu `D`.\n")
    
    initial_gini_lines = []
    initial_gini = calculate_gini([row[target_index] for row in data], initial_gini_lines)
    output.extend(initial_gini_lines)
    output.append(f"\n=> **Gini ban đầu của tập dữ liệu Gini(D)** = **{initial_gini:.3f}**\n")

    output.append("## 3. Tính Gini Index cho từng cách phân chia (Split)")
    output.append("CART tạo ra các cây nhị phân, vì vậy nó sẽ xem xét từng cách chia mỗi thuộc tính thành hai nhóm và tính toán Gini Index (chỉ số Gini có trọng số) cho mỗi cách chia đó.\n")
    output.append("### Công thức Gini Index")
    output.append("`Gini_index(D, A) = (|D_true|/|D|) * Gini(D_true) + (|D_false|/|D|) * Gini(D_false)`\n")

    results = {}
    best_overall_gini = 1
    best_overall_feature = None
    best_overall_split = None

    for i, feature_name in enumerate(feature_names):
        feature_index = i
        output.append(f"### 3.{i+1}. Phân tích thuộc tính: **'{feature_name}'**\n")
        unique_values = sorted(list(set([row[feature_index] for row in data])))
        
        for value in unique_values:
            output.append(f"#### Phân chia (Split): `{feature_name} == {value}` vs. `{feature_name} != {value}`")
            subset_true = [row for row in data if row[feature_index] == value]
            subset_false = [row for row in data if row[feature_index] != value]

            if not subset_true or not subset_false: 
                output.append("- Bỏ qua vì một trong hai tập con rỗng.\n")
                continue

            gini_true_lines = []
            gini_false_lines = []
            gini_true = calculate_gini([row[target_index] for row in subset_true], gini_true_lines, indent="  ")
            gini_false = calculate_gini([row[target_index] for row in subset_false], gini_false_lines, indent="  ")

            output.append(f"- **Nhánh Đúng (True)**: `{feature_name} == {value}` (có {len(subset_true)} mẫu)")
            output.extend(gini_true_lines)
            output.append(f"- **Nhánh Sai (False)**: `{feature_name} != {value}` (có {len(subset_false)} mẫu)")
            output.extend(gini_false_lines)

            weight_true = len(subset_true) / len(data)
            weight_false = len(subset_false) / len(data)
            gini_index = (weight_true * gini_true) + (weight_false * gini_false)
            
            output.append("- **Tính Gini Index cho phân chia này:**")
            output.append(f"  - Gini_index = (Trọng số True * Gini True) + (Trọng số False * Gini False)")
            output.append(f"  - Gini_index = ({weight_true:.3f} * {gini_true:.3f}) + ({weight_false:.3f} * {gini_false:.3f}) = **{gini_index:.3f}**\n")

            if feature_name not in results or gini_index < results[feature_name]['gini_index']:
                results[feature_name] = {'gini_index': gini_index, 'split_value': value}
            
            if gini_index < best_overall_gini:
                best_overall_gini = gini_index
                best_overall_feature = feature_name
                best_overall_split = value
    
    output.append("## 4. Tổng kết và Chọn cách phân chia tốt nhất")
    output.append("So sánh Gini Index thấp nhất từ mỗi thuộc tính.\n")
    output.append("| Thuộc tính (Feature) | Cách chia tốt nhất (Best Split) | Gini Index thấp nhất |")
    output.append("|---|---|---|")
    sorted_summary = sorted(results.items(), key=lambda item: item[1]['gini_index'])
    for feature, values in sorted_summary:
        output.append(f"| {feature} | '{values['split_value']}' vs Others | {values['gini_index']:.3f} |")
    
    output.append(f"\n**=> Kết luận:** Cách phân chia có Gini Index thấp nhất tổng thể là của thuộc tính **'{best_overall_feature}'** khi chia theo điều kiện **`{best_overall_feature} == {best_overall_split}`** (Gini Index = **{best_overall_gini:.3f}**). Đây sẽ là nút gốc của cây.\n")
    
    output.append("## 5. Xây dựng Cây quyết định (Decision Tree)")
    tree = build_cart_tree(data, feature_names)
    output.append("Cây được xây dựng bằng cách lặp lại quy trình trên cho mỗi nút con.\n")
    output.append("```")
    output.append(format_cart_tree(tree))
    output.append("```")

    return "\n".join(output)

if __name__ == "__main__":
    output_content = process_data_for_cart_detailed('data.csv')
    with open('cart_output.md', 'w', encoding='utf-8') as f:
        f.write(output_content)
