import numpy as np
import tensorflow as tf
import autokeras as ak

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import load_model
from sklearn.metrics import r2_score

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


model2 = model.export_model()
try:
    model2.save('ak_save_boston', save_format='tf')
except:
    model2.save('ak_save_boston.h5')

model3 = load_model('ak_save_boston', custom_objects=ak.CUSTOM_OBJECTS)
result_boston = model3.evaluate(x_test, y_test)

y_pred = model3.predict(x_test)
r2 = r2_score(y_test, y_pred)

print("load_result :", result_boston, r2)
