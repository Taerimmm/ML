import numpy as np
import tensorflow as tf
import autokeras as ak

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score

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

model2 = model.export_model()
try:
    model2.save('ak_save_boston', save_format='tf')
except:
    model2.save('ak_save_boston.h5')

model3 = load_model('ak_save_boston', custom_objects=ak.CUSTOM_OBJECTS)
result_boston = model3.evaluate(x_test, y_test)

y_pred = model3.predict(x_test).round()
acc = accuracy_score(y_test, y_pred)

print("load_result :", result_boston, acc)
