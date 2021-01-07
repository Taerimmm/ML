# x = np.array([3,6,5,4,2]) 이거를 sklearn의 onehotencoder와 keras의 to_categorical써서 결과치를 비교

import numpy as np
x = np.array([3,6,5,4,2])

# OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
labels = x
label_encoder = LabelEncoder()
label_ids = label_encoder.fit_transform(labels)

encoder = OneHotEncoder(sparse=False)
reshaped = label_ids.reshape(len(label_ids), 1)
x1 = encoder.fit_transform(reshaped)
print(x1)

# to_categorical
from tensorflow.keras.utils import to_categorical
x2 = to_categorical(x)
print(x2)