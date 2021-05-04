import numpy as np
import tensorflow as tf
import autokeras as ak

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataset = load_boston()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

model = ak.StructuredDataRegressor(
    overwrite=True,
    max_trials=4
)

model.fit(x_train, y_train, epochs=10)

result = model.evaluate(x_test, y_test)
print(result)