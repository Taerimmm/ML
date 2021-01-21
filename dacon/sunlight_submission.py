import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model
model_2 = load_model('./dacon/data/sunlight_model_binary.hdf5')

submission = pd.DataFrame(np.zeros((7776,9)))
submission.columns = pd.read_csv('./dacon/data/sample_submission.csv', index_col=0, header=0).columns
submission.index = pd.read_csv('./dacon/data/sample_submission.csv', index_col=0, header=0).index

data = pd.read_csv('./dacon/data/train/train.csv', header=0)

def add_features(data):
    data['cos'] = np.cos(np.pi/2 - np.abs(data['Hour']%12 - 6)/6*np.pi/2)
    data.insert(3, 'GHI', data['DNI']*data['cos']+data['DHI'])
    data.drop(['cos'], axis=1, inplace=True)
    return data

data = add_features(data).iloc[:,3:]
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(data)


'''
# 분포 확인
submission = pd.read_csv('./dacon/data/sample_submission_new.csv', index_col=0, header=0)
import matplotlib.pyplot as plt
plt.plot(submission.iloc[:,-1][:192])
plt.grid()
plt.show()
'''