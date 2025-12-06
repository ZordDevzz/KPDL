import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Load the data
data = pd.read_csv("data.csv")
X = data["height"]
y = data["weight"]

# Visualize the data
plt.scatter(X, y, label="Data")
plt.xlabel("Height (cm)")
plt.ylabel("Weight (kg)")
plt.title("Height vs Weight")
plt.legend()
plt.savefig("height_vs_weight.png")
plt.close()

# Tạo Xbar = [1, x]
X_reshaped = X.values.reshape(-1, 1)
one = np.ones((X_reshaped.shape[0], 1))
Xbar = np.concatenate((one, X_reshaped), axis=1)

# Calculate weights w using the normal equation w = (X^T * X)^(-1) * X^T * y
A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)
w = np.dot(np.linalg.inv(A), b)
print("w = ", w)

# Extract the intercept and slope from the weight vector
b_0 = w[0]
b_1 = w[1]
x0 = np.linspace(145, 185, 2, endpoint=True)
y0 = b_0 + b_1 * x0

# Visualize the data and the regression line
plt.scatter(X, y, color="blue", label="Actual data")
plt.plot(x0, y0, color="red", label="Regression line")
plt.xlabel("Height (cm)")
plt.ylabel("Weight (kg)")
plt.title("Height vs Weight with Regression Line")
plt.legend()
plt.savefig("evaluation.png")

# CLI input for height and predict weight
try:
    input_height = float(input("Enter a height in cm to predict weight: "))
    predicted_weight = b_0 + b_1 * input_height
    print(f"For a height of {input_height:.2f} cm, the predicted weight is {predicted_weight:.2f} kg.")
except ValueError:
    print("Invalid input. Please enter a numerical value for height.")

