from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt

iris_dataset = load_iris()

print(f"Keys of iris: {iris_dataset.keys()}")

print(f"Features: {iris_dataset.feature_names}")

print(iris_dataset['DESCR'][:193]+"\n")

print(f"Target Names: {iris_dataset['target_names']}")

print(f"Data set: {iris_dataset['data']}")

print(f"Data Type: {type(iris_dataset['data'])}")

print(f"Data Shape: {iris_dataset['data'].shape}")

print(f"Target: {iris_dataset['target']}")

print(f"Target: {iris_dataset['target'].shape}")

X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

print(f"X_train: {X_train.shape}")

iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset['feature_names'])

print(iris_dataframe)

gr = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o', hist_kwds={'bins': 20}, s=60, alpha=.8)
plt.show()

