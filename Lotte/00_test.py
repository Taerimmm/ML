import os
import numpy as np
import pandas as pd
import cv2

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

img = cv2.imread('../data/LPD_competition/train/0/0.jpg')
print(img.shape)

answer = pd.read_csv('./Lotte/sample.csv', header=0)
print(answer)

answer.iloc[1,1] = 1
answer.to_csv('./Lotte/submission.csv', index=False)