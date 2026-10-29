import pandas as pd
from sklearn.datasets import make_classification
import os

X, y = make_classification(
    n_samples=600,
    n_features=8,
    n_informative=5,
    n_redundant=1,
    random_state=42
)

df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(8)])
df["target"] = y

os.makedirs(os.path.join("data", "raw"), exist_ok=True)
df.to_csv(os.path.join("data", "raw", "dataset.csv"), index=False)

print("Dataset criado em data/raw/dataset.csv")
print("Shape:", df.shape)
print(df.head())
