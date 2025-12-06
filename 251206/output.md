# C4.5 Algorithm - Information Gain Calculation
## Dataset: data.csv
Total instances: 6
Features: humid, weather, wind
Target variable: 'decision'

### 1. Entropy of the Target Variable (Decision)
Formula: `Entropy(S) = - Σ (p_i * log2(p_i))`

| Class | Count | Probability (p_i) | -p_i * log2(p_i) |
|---|---|---|---|
| no | 4 | 0.667 | 0.390 |
| yes | 2 | 0.333 | 0.528 |

**Entropy(Decision)** = 0.390 + 0.528 = **0.918**

### 2. Information Gain for each Feature
Formula: `Gain(S, A) = Entropy(S) - Σ ((Count(S_v) / Count(S)) * Entropy(S_v))`

#### Feature: 'humid'
##### Sub-Entropy Calculation for 'humid'
| humid Value (v) | Count(S_v) | Class Counts | Entropy(S_v) | Weight (Count(S_v)/Count(S)) | Weighted Entropy |
|---|---|---|---|---|---|
| high | 3 | no: 2, yes: 1 | 0.918 | 0.500 | 0.459 |
| normal | 3 | no: 2, yes: 1 | 0.918 | 0.500 | 0.459 |

Sum of weighted entropies for 'humid' = 0.918
**Information Gain('humid')** = Entropy(Decision) - Σ(...)
= 0.918 - 0.918 = **0.000**

#### Feature: 'weather'
##### Sub-Entropy Calculation for 'weather'
| weather Value (v) | Count(S_v) | Class Counts | Entropy(S_v) | Weight (Count(S_v)/Count(S)) | Weighted Entropy |
|---|---|---|---|---|---|
| overcast | 1 | yes: 1 | 0.000 | 0.167 | 0.000 |
| rainy | 3 | no: 3 | 0.000 | 0.500 | 0.000 |
| sunny | 2 | no: 1, yes: 1 | 1.000 | 0.333 | 0.333 |

Sum of weighted entropies for 'weather' = 0.333
**Information Gain('weather')** = Entropy(Decision) - Σ(...)
= 0.918 - 0.333 = **0.585**

#### Feature: 'wind'
##### Sub-Entropy Calculation for 'wind'
| wind Value (v) | Count(S_v) | Class Counts | Entropy(S_v) | Weight (Count(S_v)/Count(S)) | Weighted Entropy |
|---|---|---|---|---|---|
| strong | 2 | no: 2 | 0.000 | 0.333 | 0.000 |
| weak | 4 | no: 2, yes: 2 | 1.000 | 0.667 | 0.667 |

Sum of weighted entropies for 'wind' = 0.667
**Information Gain('wind')** = Entropy(Decision) - Σ(...)
= 0.918 - 0.667 = **0.252**

### 3. Summary of Information Gains
| Feature | Information Gain |
|---|---|
| weather | 0.585 |
| wind | 0.252 |
| humid | 0.000 |

**Conclusion:** The feature with the highest Information Gain is **'weather'** with a Gain of **0.585**.