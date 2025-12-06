# C4.5 Algorithm - Gain Ratio Calculation
## Dataset: data.csv
Total instances: 6
Features: humid, weather, wind
Target variable: 'decision'

### 1. Entropy of the Target Variable (I)
Formula: `I = Entropy(S) = - Σ (p_i * log2(p_i))`

| Class | Count | Probability (p_i) | -p_i * log2(p_i) |
|---|---|---|---|
| no | 4 | 0.667 | 0.390 |
| yes | 2 | 0.333 | 0.528 |

**I = Entropy(Decision)** = 0.390 + 0.528 = **0.918**

### 2. Gain Ratio for each Feature
#### Feature: 'humid'
##### a) Information Gain (Gain)
Formula: `Gain(S, A) = I - Σ ((|S_v| / |S|) * Entropy(S_v))`

| humid Value (v) | Count | Class Counts | Entropy(S_v) | Weight | Weighted Entropy |
|---|---|---|---|---|---|
| high | 3 | no: 2, yes: 1 | 0.918 | 0.500 | 0.459 |
| normal | 3 | no: 2, yes: 1 | 0.918 | 0.500 | 0.459 |

**Gain('humid')** = 0.918 - 0.918 = **0.000**

##### b) Split Information (SI)
Formula: `SI(A) = - Σ ((|S_v| / |S|) * log2(|S_v| / |S|))`

| humid Value (v) | Count | Proportion | -p * log2(p) |
|---|---|---|---|
| high | 3 | 0.500 | 0.500 |
| normal | 3 | 0.500 | 0.500 |

**SI('humid')** = 0.500 + 0.500 = **1.000**

##### c) Gain Ratio (GR)
Formula: `GR(A) = Gain(A) / SI(A)`
**GR('humid')** = 0.000 / 1.000 = **0.000**

#### Feature: 'weather'
##### a) Information Gain (Gain)
Formula: `Gain(S, A) = I - Σ ((|S_v| / |S|) * Entropy(S_v))`

| weather Value (v) | Count | Class Counts | Entropy(S_v) | Weight | Weighted Entropy |
|---|---|---|---|---|---|
| overcast | 1 | yes: 1 | 0.000 | 0.167 | 0.000 |
| rainy | 3 | no: 3 | 0.000 | 0.500 | 0.000 |
| sunny | 2 | no: 1, yes: 1 | 1.000 | 0.333 | 0.333 |

**Gain('weather')** = 0.918 - 0.333 = **0.585**

##### b) Split Information (SI)
Formula: `SI(A) = - Σ ((|S_v| / |S|) * log2(|S_v| / |S|))`

| weather Value (v) | Count | Proportion | -p * log2(p) |
|---|---|---|---|
| overcast | 1 | 0.167 | 0.431 |
| rainy | 3 | 0.500 | 0.500 |
| sunny | 2 | 0.333 | 0.528 |

**SI('weather')** = 0.431 + 0.500 + 0.528 = **1.459**

##### c) Gain Ratio (GR)
Formula: `GR(A) = Gain(A) / SI(A)`
**GR('weather')** = 0.585 / 1.459 = **0.401**

#### Feature: 'wind'
##### a) Information Gain (Gain)
Formula: `Gain(S, A) = I - Σ ((|S_v| / |S|) * Entropy(S_v))`

| wind Value (v) | Count | Class Counts | Entropy(S_v) | Weight | Weighted Entropy |
|---|---|---|---|---|---|
| strong | 2 | no: 2 | 0.000 | 0.333 | 0.000 |
| weak | 4 | no: 2, yes: 2 | 1.000 | 0.667 | 0.667 |

**Gain('wind')** = 0.918 - 0.667 = **0.252**

##### b) Split Information (SI)
Formula: `SI(A) = - Σ ((|S_v| / |S|) * log2(|S_v| / |S|))`

| wind Value (v) | Count | Proportion | -p * log2(p) |
|---|---|---|---|
| strong | 2 | 0.333 | 0.528 |
| weak | 4 | 0.667 | 0.390 |

**SI('wind')** = 0.528 + 0.390 = **0.918**

##### c) Gain Ratio (GR)
Formula: `GR(A) = Gain(A) / SI(A)`
**GR('wind')** = 0.252 / 0.918 = **0.274**

### 3. Summary of Calculations
| Feature | Gain (G) | Split Info (SI) | Gain Ratio (GR) |
|---|---|---|---|
| weather | 0.585 | 1.459 | 0.401 |
| wind | 0.252 | 0.918 | 0.274 |
| humid | 0.000 | 1.000 | 0.000 |

**Conclusion:** The feature with the highest Gain Ratio is **'weather'** with a GR of **0.401**.