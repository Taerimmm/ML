# My_Picture Predict
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

model = load_model('../data/h5/k67_img.h5')

pred_datagen = ImageDataGenerator(rescale=1./255) 

pred_data = pred_datagen.flow_from_directory(
    '../data/image',
    classes=['my'],
    target_size=(150,150),
    batch_size=1,
    class_mode=None
)
print(pred_data[0])

pred = model.predict_generator(pred_data)
plt.imshow(pred_data[0].reshape(150,150,3))
plt.show()
print(pred)

print('======================================')
if pred > 0.5:
    print("남자 acc =", pred)
else:
    print("여자 acc =", pred)

# 남자 acc = [[0.65340704]]

img = cv2.imread('../data/image/my/my.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, dsize=(150,150)) / 255.0 
plt.imshow(img)
plt.show()
print(img)

result = model.predict(np.array([img]))
print(result)

print('======================================')
if result > 0.5:
    print("남자 acc =", result)
else:
    print("여자 acc =", result)

# 남자 acc = [[0.65340704]]
