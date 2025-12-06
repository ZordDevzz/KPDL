import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Load the data
data = pd.read_csv("Position_Salaries.csv")
X = data.iloc[:, 1:2].values
y = data.iloc[:, 2].values


degree = 4
poly_reg = PolynomialFeatures(degree=degree)
X_poly = poly_reg.fit_transform(X)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# print(f"Polynomial model coefficients: {lin_reg_2.coef_}")
# print(f"Polynomial model intercept: {lin_reg_2.intercept_}")

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
y_grid = lin_reg_2.predict(poly_reg.fit_transform(X_grid))

plt.scatter(X, y, color="blue", label="Data")
plt.plot(X_grid, y_grid, color="red", label="Polynomial regression line")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.title("Salary & Position Level with Polynomial Regression")
plt.legend()
plt.savefig("evaluation_salary_poly_sk.png")

try:
    input_level = float(input("Nhập Position Level: "))

    input_poly = poly_reg.transform([[input_level]])
    predicted_salary = lin_reg_2.predict(input_poly)

    print(f"Với level {input_level:.2f}, Salary dự đoán là: {predicted_salary[0]:.2f}.")
except ValueError:
    print("Giá trị nhập vào không hợp lệ. Hãy nhập một số thực.")
