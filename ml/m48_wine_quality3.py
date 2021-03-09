import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

wine = pd.read_csv('../data/csv/winequality-white.csv', header=0, sep=';')
print(wine.head())
print(wine.shape)   # (4898, 12)
print(wine.describe())

wine_npy = wine.to_numpy()

# x = wine_npy[:, :11]
# y = wine_npy[:, 11]

y = wine['quality']
x = wine.drop('quality', axis=1)

newlist = []
for i in list(y):
    if i <= 4:
        newlist += [0]
    elif i <= 7:
        newlist += [1]
    else:
        newlist += [2]
y = newlist

# print(x.shape, y.shape) # (4898, 11) (4898,)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size=0.8)

scale = StandardScaler()
scale.fit(x_train)
x_train = scale.transform(x_train)
x_test = scale.transform(x_test)
print(x_train.shape, x_test.shape)  # (3918, 11) (980, 11)

# model = KNeighborsClassifier()      # score : 0.5663265306122449
model = RandomForestClassifier()    # score : 0.7051020408163265 -> 0.9489795918367347
# model = XGBClassifier()             # score : 0.6816326530612244

model.fit(x_train, y_train)

score = model.score(x_test, y_test)

print('score :', score)