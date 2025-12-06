import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression

# Load the data
data = pd.read_csv("House_Price.csv")
X = data["m2"]
y = data["price"]

# Visualize the data
plt.scatter(X, y, label="Data")
plt.xlabel("Square Meters (m2)")
plt.ylabel("Price")
plt.title("House Price vs Square Meters")
plt.legend()
plt.savefig("house_price_vs_m2.png")
plt.close()


# Reshape data for sklearn
X_reshaped = X.values.reshape(-1, 1)

# Build and train the model
model = LinearRegression()
model.fit(X_reshaped, y)

# Get the predicted prices for the entire dataset
predicted_prices = model.predict(X_reshaped)

# The model's learned coefficients
b_0 = model.intercept_
b_1 = model.coef_[0]
print(f"w = [{b_0}, {b_1}]")

# Visualize the data and the regression line
plt.scatter(X, y, color="blue", label="Actual data")
plt.plot(X, predicted_prices, color="red", label="Regression line")
plt.xlabel("Square Meters (m2)")
plt.ylabel("Price")
plt.title("House Price vs Square Meters with Regression Line (Scikit-learn)")
plt.legend()
plt.savefig("evaluation_house_price.png")

# CLI input for m2 and predict price
try:
    input_m2 = float(input("Enter square meters (m2) to predict price: "))
    input_m2_reshaped = np.array([[input_m2]])
    predicted_price = model.predict(input_m2_reshaped)
    print(f"For a house of {input_m2:.2f} m2, the predicted price is {predicted_price[0]:.2f}.")
except ValueError:
    print("Invalid input. Please enter a numerical value for square meters.")
