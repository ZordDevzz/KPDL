import csv
import math
from collections import Counter

def calculate_entropy(data_column):
    """
    Calculates the entropy of a given data column (list of values).
    Entropy(S) = - Σ (p_i * log2(p_i))
    """
    if not data_column:
        return 0.0

    counts = Counter(data_column)
    total_elements = len(data_column)
    entropy = 0.0

    for count in counts.values():
        probability = count / total_elements
        if probability > 0: # Avoid log(0)
            entropy -= probability * math.log2(probability)
    return entropy

def calculate_split_info(data, feature_index):
    """
    Calculates the split information for a given feature.
    SplitInfo(S, A) = - Σ (|S_v| / |S|) * log2(|S_v| / |S|)
    """
    total_elements = len(data)
    feature_values = [row[feature_index] for row in data]
    counts = Counter(feature_values)
    split_info = 0.0

    for count in counts.values():
        proportion = count / total_elements
        if proportion > 0:
            split_info -= proportion * math.log2(proportion)
    return split_info

def process_data_for_c45(csv_file):
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        raw_data = list(reader)

    # From the CSV: weekend,humid,weather,wind,decision,
    if header and header[-1] == '':
        header = header[:-1]

    feature_names = header[1:-1]
    target_name = header[-1]

    data = []
    for row in raw_data:
        if row and row[-1] == '':
            row = row[:-1]
        data.append(row[1:])

    target_index = len(data[0]) - 1
    
    target_column = [row[target_index] for row in data]
    initial_entropy = calculate_entropy(target_column)

    output = []
    output.append("# C4.5 Algorithm - Gain Ratio Calculation")
    output.append(f"## Dataset: {csv_file}")
    output.append(f"Total instances: {len(data)}")
    output.append(f"Features: {', '.join(feature_names)}")
    output.append(f"Target variable: '{target_name}'\n")

    output.append("### 1. Entropy of the Target Variable (I)")
    output.append("Formula: `I = Entropy(S) = - Σ (p_i * log2(p_i))`\n")
    
    target_counts = Counter(target_column)
    output.append("| Class | Count | Probability (p_i) | -p_i * log2(p_i) |")
    output.append("|---|---|---|---|")
    entropy_sum_terms_text = []
    for cls, count in target_counts.items():
        prob = count / len(target_column)
        term = -prob * math.log2(prob) if prob > 0 else 0
        output.append(f"| {cls} | {count} | {prob:.3f} | {term:.3f} |")
        entropy_sum_terms_text.append(f"{term:.3f}")
    output.append(f"\n**I = Entropy(Decision)** = {" + ".join(entropy_sum_terms_text)} = **{initial_entropy:.3f}**\n")

    output.append("### 2. Gain Ratio for each Feature")
    
    results = {}
    for i, feature_name in enumerate(feature_names):
        feature_index = i
        
        output.append(f"#### Feature: '{feature_name}'")
        
        # Information Gain (G(I))
        output.append("##### a) Information Gain (Gain)")
        output.append("Formula: `Gain(S, A) = I - Σ ((|S_v| / |S|) * Entropy(S_v))`\n")
        
        unique_feature_values = sorted(list(set([row[feature_index] for row in data])))
        sum_weighted_entropy = 0.0

        output.append(f"| {feature_name} Value (v) | Count | Class Counts | Entropy(S_v) | Weight | Weighted Entropy |")
        output.append("|---|---|---|---|---|---|")
        
        for value in unique_feature_values:
            subset_data_target = [row[target_index] for row in data if row[feature_index] == value]
            subset_size = len(subset_data_target)
            
            subset_entropy = calculate_entropy(subset_data_target) # This is S
            weight = subset_size / len(data)
            weighted_term = weight * subset_entropy
            sum_weighted_entropy += weighted_term

            class_counts = Counter(subset_data_target)
            class_str = ", ".join([f"{cls}: {count}" for cls, count in sorted(class_counts.items())])
            
            output.append(f"| {value} | {subset_size} | {class_str} | {subset_entropy:.3f} | {weight:.3f} | {weighted_term:.3f} |")
            
        information_gain = initial_entropy - sum_weighted_entropy
        output.append(f"\n**Gain('{feature_name}')** = {initial_entropy:.3f} - {sum_weighted_entropy:.3f} = **{information_gain:.3f}**\n")

        # Split Information (S)
        output.append("##### b) Split Information (SI)")
        output.append("Formula: `SI(A) = - Σ ((|S_v| / |S|) * log2(|S_v| / |S|))`\n")
        split_info = calculate_split_info(data, feature_index)
        
        output.append(f"| {feature_name} Value (v) | Count | Proportion | -p * log2(p) |")
        output.append("|---|---|---|---|")
        si_sum_terms_text = []
        for value in unique_feature_values:
            count = len([row for row in data if row[feature_index] == value])
            proportion = count / len(data)
            term = -proportion * math.log2(proportion) if proportion > 0 else 0
            output.append(f"| {value} | {count} | {proportion:.3f} | {term:.3f} |")
            si_sum_terms_text.append(f"{term:.3f}")
        output.append(f"\n**SI('{feature_name}')** = {" + ".join(si_sum_terms_text)} = **{split_info:.3f}**\n")

        # Gain Ratio (GR)
        output.append("##### c) Gain Ratio (GR)")
        gain_ratio = information_gain / split_info if split_info > 0 else 0
        output.append("Formula: `GR(A) = Gain(A) / SI(A)`")
        output.append(f"**GR('{feature_name}')** = {information_gain:.3f} / {split_info:.3f} = **{gain_ratio:.3f}**\n")

        results[feature_name] = {
            'gain': information_gain,
            'split_info': split_info,
            'gain_ratio': gain_ratio
        }

    output.append("### 3. Summary of Calculations")
    output.append("| Feature | Gain (G) | Split Info (SI) | Gain Ratio (GR) |")
    output.append("|---|---|---|---|")
    
    sorted_results = sorted(results.items(), key=lambda item: item[1]['gain_ratio'], reverse=True)
    
    for feature, values in sorted_results:
        output.append(f"| {feature} | {values['gain']:.3f} | {values['split_info']:.3f} | {values['gain_ratio']:.3f} |")
    
    best_feature, best_values = sorted_results[0]
    
    output.append(f"\n**Conclusion:** The feature with the highest Gain Ratio is **'{best_feature}'** with a GR of **{best_values['gain_ratio']:.3f}**.")

    return "\n".join(output)

if __name__ == "__main__":
    output_content = process_data_for_c45('data.csv')
    with open('output.md', 'w', encoding='utf-8') as f:
        f.write(output_content)