from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

iris_dataset = load_iris()
knn = KNeighborsClassifier(n_neighbors=1)

X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

knn.fit(X_train, y_train)

y_predict = knn.predict(X_test)
print(f"Prediction: {y_predict}")

print(f"Prediction Score: {np.mean(y_predict == y_test)}")
print(f"Test Score: {knn.score(X_test, y_test)}")