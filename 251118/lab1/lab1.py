import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

Dushanbe = pd.read_csv("Dushanbe_house.csv")

print("Raw Data: \n")
print(Dushanbe.isnull().sum())


# cleaning data
Dushanbe.dropna(axis=0, inplace=True)
# if "Unnamed: 0" in Dushanbe.columns:
#     Dushanbe.drop(columns=["Unnamed: 0"], inplace=True)

print("\nCleaned Data: \n")
print(Dushanbe.isnull().sum())

# saving cleaned data to a new csv file
Dushanbe.to_csv("./Dushanbe_house_cleaned.csv", index=False)
print("\nCleaned data saved to 'Dushanbe_house_cleaned.csv'")

# separating inputs and outputs
columns = Dushanbe.columns

Inputs = Dushanbe[columns[:-1]]
outputs = Dushanbe[columns[-1]]

X_train, X_test, y_train, y_test = train_test_split(
    Inputs, outputs, test_size=0.25, random_state=42
)

model = Ridge(alpha=0.9)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# evaluating the model
print("\nModel Evaluation:")
print("R^2 Score: ", r2_score(y_test, y_pred))

# plotting the results
plt.figure(figsize=(15, 8))
plt.plot(y_test.values, label="Actual Prices", color="b")
plt.plot(y_pred, label="Predicted Prices", color="r")
plt.title("Actual vs Predicted House Prices")
plt.xlabel("Sample Index")
plt.ylabel("House Price")
plt.legend()
plt.show()