
import pandas as pd
from collections import Counter
from itertools import chain, combinations

def calculate_gini(series):
    """Tính Gini impurity cho một chuỗi (cột) dữ liệu."""
    if series.empty:
        return 0
    counts = Counter(series)
    impurity = 1
    for label in counts:
        prob_of_label = counts[label] / len(series)
        impurity -= prob_of_label**2
    return impurity

def get_attribute_score_multiway(df, attribute, target):
    """
    Bước 1: Đánh giá chất lượng thuộc tính bằng Gini đa nhánh.
    Đây là "điểm số" để so sánh các thuộc tính với nhau.
    """
    total_samples = len(df)
    if total_samples == 0:
        return 0

    weighted_gini = 0
    for value in df[attribute].unique():
        subset = df[df[attribute] == value]
        subset_gini = calculate_gini(subset[target])
        weighted_gini += (len(subset) / total_samples) * subset_gini
    return weighted_gini

def find_best_binary_split(df, attribute, target):
    """
    Bước 2: Tìm cách chia nhị phân tốt nhất cho một thuộc tính đã được chọn.
    """
    best_split = None
    min_weighted_gini = 1
    
    values = df[attribute].unique().tolist()
    if len(values) <= 1:
        return None, min_weighted_gini

    # Tạo tất cả các cách chia đôi có thể
    # Ví dụ: [a,b,c] -> ({a}, {b,c}), ({b}, {a,c}), ({c}, {a,b})
    power_set = chain.from_iterable(combinations(values, r) for r in range(1, len(values)//2 + 1))

    for group1_tuple in power_set:
        group1 = set(group1_tuple)
        group2 = set(values) - group1

        df_group1 = df[df[attribute].isin(group1)]
        df_group2 = df[df[attribute].isin(group2)]

        gini1 = calculate_gini(df_group1[target])
        gini2 = calculate_gini(df_group2[target])

        weighted_gini = (len(df_group1) / len(df)) * gini1 + (len(df_group2) / len(df)) * gini2

        if weighted_gini < min_weighted_gini:
            min_weighted_gini = weighted_gini
            best_split = (list(group1), list(group2))

    return best_split, min_weighted_gini

def build_tree(df, features, target, indent=""):
    """Hàm đệ quy để xây dựng và in cây quyết định."""
    
    # Điều kiện dừng 1: Nếu tất cả các mẫu đều thuộc một lớp (nút thuần khiết)
    if len(df[target].unique()) == 1:
        print(f"{indent}└── Kết quả: {df[target].iloc[0]} (Nút lá thuần khiết)")
        return

    # Điều kiện dừng 2: Nếu không còn thuộc tính nào để chia
    if not features:
        majority_class = df[target].mode()[0]
        print(f"{indent}└── Kết quả: {majority_class} (Hết thuộc tính, chọn lớp đa số)")
        return
        
    print(f"{indent}--- Phân tích nút (Số mẫu: {len(df)}) ---")
    print(f"{indent}Các thuộc tính đang xét: {features}")
    
    # Bước 1: Đánh giá chất lượng của các thuộc tính
    best_attribute = None
    min_score = 1
    scores = {}
    for feature in features:
        score = get_attribute_score_multiway(df, feature, target)
        scores[feature] = round(score, 4)
        if score < min_score:
            min_score = score
            best_attribute = feature
    
    print(f"{indent}Đánh giá Gini đa nhánh: {scores}")
    print(f"{indent}=> Thuộc tính tốt nhất được chọn: '{best_attribute}' (Gini đa nhánh = {round(min_score, 4)})")

    # Bước 2: Tìm cách chia nhị phân tốt nhất cho thuộc tính đã chọn
    best_split_groups, split_gini = find_best_binary_split(df, best_attribute, target)
    
    if best_split_groups is None:
        majority_class = df[target].mode()[0]
        print(f"{indent}└── Không thể chia tiếp. Kết quả: {majority_class}")
        return

    print(f"{indent}Thực hiện chia nhị phân trên '{best_attribute}': {best_split_groups[0]} vs {best_split_groups[1]} (Gini trọng số = {round(split_gini, 4)})")
    print("")

    # Phân nhánh và đệ quy
    remaining_features = [f for f in features if f != best_attribute]
    
    group1 = best_split_groups[0]
    df_group1 = df[df[best_attribute].isin(group1)]
    print(f"{indent}Nhánh 1: Nếu `{best_attribute}` là `{group1}` ({len(df_group1)} mẫu)")
    build_tree(df_group1, remaining_features, target, indent + "|  ")
    print("")

    group2 = best_split_groups[1]
    df_group2 = df[df[best_attribute].isin(group2)]
    print(f"{indent}Nhánh 2: Nếu `{best_attribute}` là `{group2}` ({len(df_group2)} mẫu)")
    build_tree(df_group2, remaining_features, target, indent + "|  ")


# --- Main ---
if __name__ == "__main__":
    file_path = 'data.csv'
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file '{file_path}'. Vui lòng đảm bảo file nằm cùng thư mục.")
        exit()

    # Xác định các thuộc tính và biến mục tiêu
    # Giả sử 'Weekend' là cột ID, 'Decision' là cột mục tiêu.
    # Các cột còn lại là thuộc tính.
    if 'Weekend' in df.columns:
        features = [col for col in df.columns if col not in ['Weekend', 'Decision']]
    else:
        features = [col for col in df.columns if col != 'Decision']
    
    target = 'Decision'

    if target not in df.columns:
        print(f"Lỗi: Không tìm thấy cột mục tiêu '{target}' trong file '{file_path}'.")
        exit()
    if not features:
        print("Lỗi: Không tìm thấy thuộc tính nào để phân tích.")
        exit()

    print(f"Đọc dữ liệu từ '{file_path}'...")
    print("Các thuộc tính được sử dụng:", features)
    print("Biến mục tiêu:", target)
    print("\nBắt đầu xây dựng cây quyết định từ dữ liệu gốc...")
    build_tree(df, features, target)
