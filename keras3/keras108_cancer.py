import numpy as np
import tensorflow as tf
import autokeras as ak

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = ak.StructuredDataClassifier(
    overwrite=True,
    max_trials=2
)

model.fit(x_train, y_train, epochs=10)

result = model.evaluate(x_test, y_test)
print(result)