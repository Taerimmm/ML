# Predict
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

test_data = pd.read_csv('./dacon2/data/test.csv', index_col=0, header=0)
x_test = test_data.drop(['letter'], axis=1).values
x_test = x_test.reshape(-1, 28, 28, 1)
x_test = x_test/255

datagen = ImageDataGenerator(
    width_shift_range=(-1,1),  
    height_shift_range=(-1,1))
datagen2 = ImageDataGenerator()

# Case 1 : best model selection

''' 모든 모델에 적용가능한 것으로 만들 것 '''
i_max = 7 # 따로 설정해줄것 !! 7 --> ?

print('Best Model is {}\'s'.format(i_max))
model = load_model('./dacon2/data/vision_model_{}.hdf5'.format(i_max))


submission = pd.read_csv('./dacon2/data/submission.csv', index_col=0, header=0)

submission['digit'] = np.argmax(model.predict(x_test), axis=1)
print(submission)

submission.to_csv('./dacon2/data/submission_model_best.csv')


# Case 2 : KFold 값 평균내기
submission2 = pd.read_csv('./dacon2/data/submission.csv', index_col=0, header=0)

steps = 40
result = 0
for i in range(steps):
    model = load_model('./dacon2/data/vision_model_{}.hdf5'.format(i))

    result += model.predict_generator(datagen2.flow(x_test, shuffle=False)) / steps
    submission2['digit'] = result.argmax(1)

print(submission2)
submission2.to_csv('./dacon2/data/submission_model_mean.csv')
