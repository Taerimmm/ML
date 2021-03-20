import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# Predict
x_test = np.load('../data/LPD_competition/npy/test_data.npy')
print(x_test.shape)


test_generator = ImageDataGenerator(rescale=1./255).flow(x_test, shuffle=False)

result = 0
for i in range(5):
    model = load_model('./dacon3/data/vision_2_model_{}.hdf5'.format(i))

    result += model.predict(test_generator) / 5


answer = pd.read_csv('./Lotte/sample.csv', header=0)
print(answer.shape)

answer.iloc[:,1] = np.argmax(result,1)
print(answer)
answer.to_csv('./Lotte/kfold_submission.csv', index=False)
