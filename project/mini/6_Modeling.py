import numpy as np
import pandas as pd

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, MaxPooling2D, Dropout


data = pd.read_csv('./project/mini/data/total_genres_mfcc.csv', header=0)

label_index = dict