import pandas as pd
from sklearn.datasets import (
    load_breast_cancer,
    load_iris,
    load_diabetes,
    load_wine,
    load_digits,
)

datasets = [
    {"name": "Breast Cancer", "function": load_breast_cancer},
    {"name": "Iris", "function": load_iris},
    {"name": "Diabetes", "function": load_diabetes},
    {"name": "Wine", "function": load_wine},
    {"name": "Digits", "function": load_digits},
]


def dataset_to_csv(dataset, filename="dataset.csv"):
    data = dataset()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    df.to_csv(filename, index=False)
    print(f"Dataset saved to {filename}")


if __name__ == "__main__":
    for ds in datasets:
        filename = f"{ds['name'].lower().replace(' ', '_')}.csv"
        dataset_to_csv(ds["function"], filename)
