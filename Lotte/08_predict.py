import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import preprocess_input

# Predict
x_test = np.load('../data/LPD_competition/npy/test_data.npy')
print(x_test.shape)

x_test = preprocess_input(x_test)

print(x_test)

result = 0
for i in range(5):
    model = load_model('./Lotte/b7_model_{}.hdf5'.format(i))

    pred = model.predict(x_test)
    print(pred)
    result += pred / 5
    # break

answer = pd.read_csv('./Lotte/sample.csv', header=0)
print(answer.shape)

answer.iloc[:,1] = np.argmax(result,1)
print(answer)
answer.to_csv('./Lotte/kfold_submission.csv', index=False)
