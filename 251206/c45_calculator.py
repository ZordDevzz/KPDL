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

def process_data_for_c45(csv_file):
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        raw_data = list(reader)

    # From the CSV: weekend,humid,weather,wind,decision,
    # This means:
    # Index 0: weekend (ID - ignored)
    # Index 1: humid (Feature)
    # Index 2: weather (Feature)
    # Index 3: wind (Feature)
    # Index 4: decision (Target)
    
    # Filter out the trailing empty column name, if it exists
    if header and header[-1] == '':
        header = header[:-1]

    # Correctly identify feature and target names
    feature_names = header[1:-1] # 'humid', 'weather', 'wind'
    target_name = header[-1] # 'decision'

    # Filter out the 'weekend' column and the trailing empty column value from data
    data = []
    for row in raw_data:
        if row and row[-1] == '':
            row = row[:-1]
        data.append(row[1:]) # Skip 'weekend' column (index 0)

    target_index = len(data[0]) - 1 # Last column is the target
    
    target_column = [row[target_index] for row in data]
    initial_entropy = calculate_entropy(target_column)

    output = []
    output.append("# C4.5 Algorithm - Information Gain Calculation")
    output.append(f"## Dataset: {csv_file}")
    output.append(f"Total instances: {len(data)}")
    output.append(f"Features: {', '.join(feature_names)}")
    output.append(f"Target variable: '{target_name}'\n")

    # Overall target entropy calculation
    output.append("### 1. Entropy of the Target Variable (Decision)")
    output.append("Formula: `Entropy(S) = - Σ (p_i * log2(p_i))`\n")
    
    target_counts = Counter(target_column)
    output.append("| Class | Count | Probability (p_i) | -p_i * log2(p_i) |")
    output.append("|---|---|---|---|")
    entropy_sum_terms_text = []
    for cls, count in target_counts.items():
        prob = count / len(target_column)
        term = -prob * math.log2(prob) if prob > 0 else 0
        output.append(f"| {cls} | {count} | {prob:.3f} | {term:.3f} |")
        entropy_sum_terms_text.append(f"{term:.3f}")
    output.append(f"\n**Entropy(Decision)** = {" + ".join(entropy_sum_terms_text)} = **{initial_entropy:.3f}**\n")

    output.append("### 2. Information Gain for each Feature")
    output.append("Formula: `Gain(S, A) = Entropy(S) - Σ ((Count(S_v) / Count(S)) * Entropy(S_v))`\n")

    gains = {}
    for i, feature_name in enumerate(feature_names):
        feature_index = i
        
        output.append(f"#### Feature: '{feature_name}'")
        
        feature_values = [row[feature_index] for row in data]
        unique_feature_values = sorted(list(set(feature_values)))
        
        sum_weighted_entropy = 0.0

        output.append(f"##### Sub-Entropy Calculation for '{feature_name}'")
        output.append(f"| {feature_name} Value (v) | Count(S_v) | Class Counts | Entropy(S_v) | Weight (Count(S_v)/Count(S)) | Weighted Entropy |")
        output.append("|---|---|---|---|---|---|")
        
        for value in unique_feature_values:
            subset_data_target = [row[target_index] for row in data if row[feature_index] == value]
            subset_size = len(subset_data_target)
            
            subset_entropy = calculate_entropy(subset_data_target)
            weight = subset_size / len(data)
            weighted_term = weight * subset_entropy
            sum_weighted_entropy += weighted_term

            class_counts = Counter(subset_data_target)
            class_str = ", ".join([f"{cls}: {count}" for cls, count in sorted(class_counts.items())])
            
            output.append(f"| {value} | {subset_size} | {class_str} | {subset_entropy:.3f} | {weight:.3f} | {weighted_term:.3f} |")
            
        output.append(f"\nSum of weighted entropies for '{feature_name}' = {sum_weighted_entropy:.3f}")
        information_gain = initial_entropy - sum_weighted_entropy
        gains[feature_name] = information_gain
        output.append(f"**Information Gain('{feature_name}')** = Entropy(Decision) - Σ(...)")
        output.append(f"= {initial_entropy:.3f} - {sum_weighted_entropy:.3f} = **{information_gain:.3f}**\n")
    
    output.append("### 3. Summary of Information Gains")
    output.append("| Feature | Information Gain |")
    output.append("|---|---|")
    
    # Sort gains for consistent output
    sorted_gains = sorted(gains.items(), key=lambda item: item[1], reverse=True)
    
    for feature, gain in sorted_gains:
        output.append(f"| {feature} | {gain:.3f} |")
    
    best_feature, max_gain = sorted_gains[0]
    
    output.append(f"\n**Conclusion:** The feature with the highest Information Gain is **'{best_feature}'** with a Gain of **{max_gain:.3f}**.")

    return "\n".join(output)

if __name__ == "__main__":
    output_content = process_data_for_c45('data.csv')
    with open('output.md', 'w', encoding='utf-8') as f:
        f.write(output_content)