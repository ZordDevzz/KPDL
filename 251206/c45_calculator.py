import csv
import math
from collections import Counter

def calculate_entropy(data_column, output_lines=None, indent=""):
    """
    Calculates the entropy of a given data column, optionally printing steps.
    """
    if not data_column:
        return 0.0

    counts = Counter(data_column)
    total_elements = len(data_column)
    entropy = 0.0
    
    if output_lines is not None:
        output_lines.append(f"{indent}- **Tính toán Entropy:**")
        
    for cls, count in counts.items():
        probability = count / total_elements
        if probability > 0:
            term = -probability * math.log2(probability)
            entropy += term
            if output_lines is not None:
                output_lines.append(f"{indent}  - P({cls}) = {count}/{total_elements} = {probability:.3f}")
                output_lines.append(f"{indent}  - Term: -({probability:.3f} * log2({probability:.3f})) = {term:.3f}")
    
    if output_lines is not None:
        sum_terms = " + ".join([f"{(-c/total_elements * math.log2(c/total_elements)):.3f}" for c in counts.values() if c > 0])
        output_lines.append(f"{indent}- **Entropy** = {sum_terms} = **{entropy:.3f}**")

    return entropy

def calculate_split_info(data, feature_index):
    total_elements = len(data)
    feature_values = [row[feature_index] for row in data]
    counts = Counter(feature_values)
    split_info = 0.0
    for count in counts.values():
        proportion = count / total_elements
        if proportion > 0:
            split_info -= proportion * math.log2(proportion)
    return split_info

def find_best_feature(data, feature_names):
    if not data or not feature_names: return None, -1
    target_index = len(data[0]) - 1
    target_column = [row[target_index] for row in data]
    initial_entropy = calculate_entropy(target_column)
    best_gain_ratio = -1
    best_feature_name = None
    for i, feature_name in enumerate(feature_names):
        feature_index = i
        sum_weighted_entropy = 0.0
        unique_feature_values = sorted(list(set([row[feature_index] for row in data])))
        for value in unique_feature_values:
            subset_data_target = [row[target_index] for row in data if row[feature_index] == value]
            subset_entropy = calculate_entropy(subset_data_target)
            weight = len(subset_data_target) / len(data)
            sum_weighted_entropy += weight * subset_entropy
        information_gain = initial_entropy - sum_weighted_entropy
        split_info = calculate_split_info(data, feature_index)
        gain_ratio = information_gain / split_info if split_info > 0 else 0
        if gain_ratio > best_gain_ratio:
            best_gain_ratio = gain_ratio
            best_feature_name = feature_name
    return best_feature_name, best_gain_ratio

def build_tree(data, feature_names):
    target_index = len(data[0]) - 1
    target_column = [row[target_index] for row in data]
    if len(set(target_column)) == 1: return target_column[0]
    if not feature_names: return Counter(target_column).most_common(1)[0][0]
    best_feature_name, _ = find_best_feature(data, feature_names)
    if best_feature_name is None: return Counter(target_column).most_common(1)[0][0]
    tree = {best_feature_name: {}}
    feature_index = feature_names.index(best_feature_name)
    remaining_features = [f for f in feature_names if f != best_feature_name]
    unique_values = sorted(list(set([row[feature_index] for row in data])))
    for value in unique_values:
        subset_data = [row for row in data if row[feature_index] == value]
        if not subset_data: tree[best_feature_name][value] = Counter(target_column).most_common(1)[0][0]
        else: tree[best_feature_name][value] = build_tree(subset_data, remaining_features)
    return tree

def format_tree(tree, indent=""):
    if not isinstance(tree, dict):
        return f"{indent}└── **Quyết định (Decision): {tree}**\n"
    lines = []
    root_feature = list(tree.keys())[0]
    for i, (value, subtree) in enumerate(tree[root_feature].items()):
        is_last = i == len(tree[root_feature]) - 1
        connector = "└──" if is_last else "├──"
        lines.append(f"{indent}{connector} Nếu (If) **{root_feature} = {value}**:\n")
        lines.append(format_tree(subtree, indent + ("    " if is_last else "│   ")))
    return "".join(lines)

def process_data_for_c45_detailed(csv_file):
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
    target_column = [row[target_index] for row in data]
    
    output = []
    output.append("# Phân tích chi tiết thuật toán C4.5")
    output.append("Đây là tài liệu hướng dẫn từng bước về cách hoạt động của thuật toán C4.5 cho mục đích giáo dục.\n")
    output.append(f"## 1. Tổng quan về Tập dữ liệu (Dataset Overview)")
    output.append(f"- **Tập dữ liệu:** `{csv_file}`")
    output.append(f"- **Tổng số mẫu:** {len(data)}")
    output.append(f"- **Các thuộc tính (Features):** {', '.join(feature_names)}")
    output.append(f"- **Thuộc tính quyết định (Target):** '{target_name}'\n")

    output.append("## 2. Tính Entropy cho toàn bộ tập dữ liệu")
    output.append("Entropy (độ hỗn loạn) đo lường sự không chắc chắn trong tập dữ liệu. Giá trị càng cao, dữ liệu càng hỗn loạn.\n")
    output.append("### Công thức Entropy (I)")
    output.append("`I = Entropy(S) = - Σ (p_i * log2(p_i))`")
    output.append("- `S`: Tập dữ liệu hiện tại.")
    output.append("- `p_i`: Tỷ lệ của lớp `i` trong tập dữ liệu `S`.\n")
    
    initial_entropy = calculate_entropy(target_column, output, indent="")
    output.append(f"\n=> **Entropy ban đầu của tập dữ liệu I(S)** = **{initial_entropy:.3f}**\n")

    output.append("## 3. Tính Gain Ratio cho từng thuộc tính")
    output.append("C4.5 sử dụng Gain Ratio để chọn thuộc tính tốt nhất để phân chia cây. Nó giải quyết vấn đề của Information Gain khi có các thuộc tính với nhiều giá trị khác nhau.\n")

    results = {}
    for i, feature_name in enumerate(feature_names):
        feature_index = i
        output.append(f"### 3.{i+1}. Phân tích thuộc tính: **'{feature_name}'**\n")
        
        # a) Information Gain
        output.append("#### a) Tính Lợi ích Thông tin (Information Gain)")
        output.append("Công thức: `Gain(S, A) = Entropy(S) - Σ ((|S_v| / |S|) * Entropy(S_v))`\n")
        
        unique_feature_values = sorted(list(set([row[feature_index] for row in data])))
        sum_weighted_entropy = 0.0
        
        entropy_s_v_calcs = []
        for value in unique_feature_values:
            subset_data_target = [row[target_index] for row in data if row[feature_index] == value]
            subset_entropy_lines = []
            subset_entropy = calculate_entropy(subset_data_target, subset_entropy_lines, indent="    ")
            
            weight = len(subset_data_target) / len(data)
            weighted_term = weight * subset_entropy
            sum_weighted_entropy += weighted_term
            
            entropy_s_v_calcs.append(f"  - **Với giá trị '{value}'** (có {len(subset_data_target)} mẫu):")
            entropy_s_v_calcs.extend(subset_entropy_lines)
            entropy_s_v_calcs.append(f"    - Trọng số (Weight) = {len(subset_data_target)}/{len(data)} = {weight:.3f}")
            entropy_s_v_calcs.append(f"    - Entropy có trọng số = {weight:.3f} * {subset_entropy:.3f} = {weighted_term:.3f}\n")

        output.append("- **Tính Entropy cho từng giá trị con (Entropy(S_v)):**")
        output.extend(entropy_s_v_calcs)
        
        information_gain = initial_entropy - sum_weighted_entropy
        output.append(f"- **Tổng Entropy có trọng số** = **{sum_weighted_entropy:.3f}**")
        output.append(f"- **Gain('{feature_name}')** = {initial_entropy:.3f} - {sum_weighted_entropy:.3f} = **{information_gain:.3f}**\n")

        # b) Split Information
        output.append("#### b) Tính Thông tin Phân tách (Split Information - SI)")
        output.append("Công thức: `SI(A) = - Σ ((|S_v| / |S|) * log2(|S_v| / |S|))`\n")
        split_info = calculate_split_info(data, feature_index)
        
        si_terms = []
        for value in unique_feature_values:
            count = len([row for row in data if row[feature_index] == value])
            proportion = count / len(data)
            term = -proportion * math.log2(proportion) if proportion > 0 else 0
            output.append(f"- **Với giá trị '{value}'**: -({count}/{len(data)}) * log2({count}/{len(data)}) = {term:.3f}")
            si_terms.append(f"{term:.3f}")
        output.append(f"- **SI('{feature_name}')** = {' + '.join(si_terms)} = **{split_info:.3f}**\n")

        # c) Gain Ratio
        output.append("#### c) Tính Tỷ lệ Lợi ích (Gain Ratio - GR)")
        gain_ratio = information_gain / split_info if split_info > 0 else 0
        output.append("Công thức: `GR(A) = Gain(A) / SI(A)`")
        output.append(f"- **GR('{feature_name}')** = {information_gain:.3f} / {split_info:.3f} = **{gain_ratio:.3f}**\n")

        results[feature_name] = { 'gain': information_gain, 'split_info': split_info, 'gain_ratio': gain_ratio }

    output.append("## 4. Tổng kết và Chọn thuộc tính tốt nhất")
    output.append("| Thuộc tính (Feature) | Gain (G) | Split Info (SI) | Gain Ratio (GR) |")
    output.append("|---|---|---|---|")
    sorted_results = sorted(results.items(), key=lambda item: item[1]['gain_ratio'], reverse=True)
    for feature, values in sorted_results:
        output.append(f"| {feature} | {values['gain']:.3f} | {values['split_info']:.3f} | {values['gain_ratio']:.3f} |")
    best_feature, best_values = sorted_results[0]
    output.append(f"\n**=> Kết luận:** Thuộc tính có Gain Ratio cao nhất là **'{best_feature}'** (GR = {best_values['gain_ratio']:.3f}). Do đó, nó sẽ được chọn làm nút gốc của cây quyết định.\n")
    
    output.append("## 5. Xây dựng Cây quyết định (Decision Tree)")
    tree = build_tree(data, feature_names)
    root_feature = list(tree.keys())[0]
    output.append(f"Nút gốc (Root): **{root_feature}**\n")
    output.append("```")
    output.append(format_tree(tree))
    output.append("```")

    return "\n".join(output)

if __name__ == "__main__":
    output_content = process_data_for_c45_detailed('data.csv')
    with open('c45_output.md', 'w', encoding='utf-8') as f:
        f.write(output_content)