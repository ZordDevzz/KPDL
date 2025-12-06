import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the data
data = pd.read_csv("data.csv")

# Visualize the data
plt.scatter(data["height"], data["weight"], label="Data")
plt.xlabel("Height (cm)")
plt.ylabel("Weight (kg)")
plt.title("Height vs Weight")
plt.legend()
plt.savefig("height_vs_weight.png")
plt.close()


# Build and train the model
model = LinearRegression()
model.fit(data[["height"]], data["weight"])

# Get the predicted weights
predicted_weights = model.predict(data[["height"]])

# Visualize the data and the model
plt.scatter(data["height"], data["weight"], color="blue", label="Actual data")
plt.plot(data["height"], predicted_weights, color="red", label="Regression line")
plt.xlabel("Height (cm)")
plt.ylabel("Weight (kg)")
plt.title("Height vs Weight with Regression Line")
plt.legend()
plt.savefig("evaluation.png")

print(
    f"Predic value of weight for height {data['height'][0]}cm is {predicted_weights[0]:.2f}kg"
)
