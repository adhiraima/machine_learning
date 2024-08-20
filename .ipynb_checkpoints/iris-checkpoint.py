from sklearn.datasets import load_iris

iris_dataset = load_iris()

print(f"Keys of iris: {iris_dataset.keys()}")

print(f"Features: {iris_dataset.feature_names}")

print(iris_dataset['DESCR'][:193]+"\n")

print(f"Target Names: {iris_dataset['target_names']}")

print(f"Data set: {iris_dataset['data']}")

print(f"Data Type: {type(iris_dataset['data'])}")

print(f"Data Shape: {iris_dataset['data'].shape}")

print(f"Target: {iris_dataset['t']}")

