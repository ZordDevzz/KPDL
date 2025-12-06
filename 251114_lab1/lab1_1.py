import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


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


# Reshape data for sklearn
# sklearn expects X to be a 2D array, so we reshape it from a 1D array to a 2D array
X_reshaped = X.values.reshape(-1, 1)

# Build and train the model
model = LinearRegression()
model.fit(X_reshaped, y)

# Get the predicted weights for the entire dataset
predicted_weights = model.predict(X_reshaped)

# The model's learned coefficients
b_0 = model.intercept_
b_1 = model.coef_[0]
print(f"w = [{b_0}, {b_1}]")

# Visualize the data and the regression line
plt.scatter(X, y, color="blue", label="Actual data")
plt.plot(X, predicted_weights, color="red", label="Regression line")
plt.xlabel("Height (cm)")
plt.ylabel("Weight (kg)")
plt.title("Height vs Weight with Regression Line (Scikit-learn)")
plt.legend()
plt.savefig("evaluation.png")

# CLI input for height and predict weight
try:
    input_height = float(input("Enter a height in cm to predict weight: "))
    # We need to reshape the input to a 2D array for the model's predict method
    input_height_reshaped = np.array([[input_height]])
    predicted_weight = model.predict(input_height_reshaped)
    print(f"For a height of {input_height:.2f} cm, the predicted weight is {predicted_weight[0]:.2f} kg.")
except ValueError:
    print("Invalid input. Please enter a numerical value for height.")

