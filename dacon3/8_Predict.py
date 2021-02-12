import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

test_datagen = ImageDataGenerator(
    rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    '../data', 
    classes=['test_dirty_mnist_2nd'],
    batch_size=5000, 
    target_size=(64, 64), 
    color_mode='grayscale',
    class_mode=None,
    shuffle=False)

for i in test_generator:
    print(i.shape)
    x_test = i
    break

submission = pd.read_csv('./dacon3/data/sample_submission.csv', index_col=0, header=0)

result = 0
steps = 1
for i in range(steps):
    model = load_model('./dacon3/data/vision_2_model_{}.hdf5'.format(i))

    result += model.predict(x_test) / steps
    print(result)
print(np.array(result).shape)
submission += np.array(result)
print(submission)

for i in range(len(submission.columns)):
    submission.iloc[:,i] = np.where(submission.iloc[:,i] > 0.5, 1, 0)
print(submission)
submission.to_csv('./dacon3/data/submission_vgg16.csv')
