import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression


# Load the data
data = pd.read_csv("Position_Salaries.csv")
X = data["Level"]
y = data["Salary"]

# Visualize the data
plt.scatter(X, y, label="Data")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.title("Salary vs Position Level")
plt.legend()
plt.savefig("salary_vs_level.png")
plt.close()


# Reshape data for sklearn
X_reshaped = X.values.reshape(-1, 1)

# Build and train the model
model = LinearRegression()
model.fit(X_reshaped, y)

# Get the predicted salaries for the entire dataset
predicted_salaries = model.predict(X_reshaped)

# The model's learned coefficients
b_0 = model.intercept_
b_1 = model.coef_[0]
print(f"w = [{b_0}, {b_1}]")

# Visualize the data and the regression line
plt.scatter(X, y, color="blue", label="Actual data")
plt.plot(X, predicted_salaries, color="red", label="Regression line")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.title("Salary vs Position Level with Regression Line (Scikit-learn)")
plt.legend()
plt.savefig("evaluation_salary.png")

# CLI input for level and predict salary
try:
    input_level = float(input("Enter a position level to predict salary: "))
    input_level_reshaped = np.array([[input_level]])
    predicted_salary = model.predict(input_level_reshaped)
    print(f"For a level of {input_level:.2f}, the predicted salary is {predicted_salary[0]:.2f}.")
except ValueError:
    print("Invalid input. Please enter a numerical value for the level.")
