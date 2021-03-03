# 이미지는 
# 구글에서 개, 고양이, 라이언, 수트 사진 1개씩 받아서
# ../data/image/vgg/ 에 4개 넣으시오

# 파일명 : dog1.jpg, cat1.jpg, ryan1.jpg, suit1.jpg

from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

img_dog = load_img('../data/image/vgg/dog1.png', target_size=(224,224))
img_cat = load_img('../data/image/vgg/cat1.jpg', target_size=(224,224))
img_ryan = load_img('../data/image/vgg/ryan1.jpg', target_size=(224,224))
img_suit = load_img('../data/image/vgg/suit1.jpg', target_size=(224,224))

# print(img_dog)  # <PIL.Image.Image image mode=RGB size=224x224 at 0x202340A9370>

# plt.imshow(img_dog)
# plt.show()

arr_dog = img_to_array(img_dog)
arr_cat = img_to_array(img_cat)
arr_ryan = img_to_array(img_ryan)
arr_suit = img_to_array(img_suit)

print(arr_dog)
print(type(arr_dog))    # <class 'numpy.ndarray'>
print(arr_dog.shape)    # (224, 224, 3)

# RGB -> BGR
from tensorflow.keras.applications.vgg16 import preprocess_input
arr_dog = preprocess_input(arr_dog)
arr_cat = preprocess_input(arr_cat)
arr_ryan = preprocess_input(arr_ryan)
arr_suit = preprocess_input(arr_suit)

print(arr_dog)
print(arr_dog.shape)    # (224, 224, 3)

arr_input = np.stack([arr_dog, arr_cat, arr_ryan, arr_suit])
print(arr_input.shape)

# 2. 모델 구성
model = VGG16()     # (4, 224, 224, 3)

results = model.predict(arr_input)
print(results)
print('results.shape :', results.shape)

# 이미지 결과 확인
from tensorflow.keras.applications.vgg16 import decode_predictions

decode_results = decode_predictions(results)
print('====================================================================================')
print('results[0] :', decode_results[0])
print('====================================================================================')
print('results[1] :', decode_results[1])
print('====================================================================================')
print('results[2] :', decode_results[2])
print('====================================================================================')
print('results[3] :', decode_results[3])
print('====================================================================================')
