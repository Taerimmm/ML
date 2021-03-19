import os
import numpy as np
import pandas as pd
from PIL import Image

# labels = os.listdir('../data/LPD_competition/train')
# print(labels)

img = []
labels = []

for dir in os.scandir('../data/LPD_competition/train'):
    label =  int(os.path.basename(dir))

    for file in os.scandir(dir):
        path = os.path.abspath(file)
        image = Image.open(path)
        image = image.resize((128,128))
        image = np.array(image)
        img.append(image)

        labels.append(label)

train = np.array(img)
labels = np.array(labels)

print(train.shape)
print(labels.shape)

np.save('../data/LPD_competition/npy/train_data.npy', arr=train)
np.save('../data/LPD_competition/npy/label_data.npy', arr=labels)


img = []

for dir in sorted(os.listdir('../data/LPD_competition/test'), key=lambda name:int(''.join(filter(str.isdigit, name)))):
    path = '../data/LPD_competition/test/' + dir
    image = Image.open(path)
    image = image.resize((128,128))
    image = np.array(image)
    img.append(image)

test = np.array(img)

print(test.shape)

np.save('../data/LPD_competition/npy/test_data.npy', arr=test)