# 실습

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectFromModel
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

wine = pd.read_csv('../data/csv/winequality-white.csv', header=0, sep=';')
print(wine.head())
print(wine.shape)   # (4898, 12)
print(wine.describe())

wine_npy = wine.to_numpy()

x = wine_npy[:, :11]
y = wine_npy[:, 11]

print(x.shape, y.shape) # (4898, 11) (4898,)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=66, shuffle=True, train_size=0.8)

scale = StandardScaler()
scale.fit(x_train)
x_train = scale.transform(x_train)
x_test = scale.transform(x_test)
print(x_train.shape, x_test.shape)  # (3918, 11) (980, 11)

model = KNeighborsClassifier()      # score : 0.5663265306122449
model = RandomForestClassifier()    # score : 0.7051020408163265
model = XGBClassifier()             # score : 0.6816326530612244

model.fit(x_train, y_train)

score = model.score(x_test, y_test)

print('score :', score)

'''
print(wine.iloc[:,-1].value_counts())

x, y = wine.iloc[:,:-1], wine.iloc[:,-1]
print(x.shape, y.shape)

x = StandardScaler().fit_transform(x)

# ohe = OneHotEncoder()
# y = ohe.fit_transform(np.array(y).reshape(-1,1)).toarray()
# print(y)
# print(y.shape)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

model = RandomForestClassifier(n_estimators=1000, random_state=0)

model.fit(x_train, y_train)

score = model.score(x_test, y_test)

print("Acc :", score)


model = XGBClassifier(n_jobs=-1)

param = [
    {"n_estimators":[100,200,300], "learning_rate":[0.001,0.01,0.1,0.3], "max_depth":[4,5,6]},
    {"n_estimators":[90,100,110], "learning_rate":[0.001,0.01,0.1], "max_depth":[4,5,6], "colsample_bytree":[0.6,0.9,1]},
    {"n_estimators":[90,110], "learning_rate":[0.001,0.1,0.5], "max_depth":[4,5,6], "colsample_bytree":[0.6,0.9,1], "colsample_bylevel":[0.6,0.7,0.9]},    
]

r2 = []
CV = [GridSearchCV, RandomizedSearchCV]
for search in CV:
    print(search.__name__)
    model_ = search(model, param)

    model_.fit(x_train, y_train, eval_metric='mlogloss')
    
    print('Acc :', model_.score(x_test, y_test))
    r2.append(model_.score(x_test, y_test))

    best_model = model_.best_estimator_
    
    thresholds = np.sort(model_.best_estimator_.feature_importances_)
    print(thresholds)
    
    print('=======================')
    break

# kfold = KFold(n_splits=5, shuffle=True)

# acc = []
# for train_idx, test_idx in kfold.split(x,y):
#     x_train, x_test = x.iloc[train_idx,:], x.iloc[test_idx,:]
#     y_train, y_test = y[train_idx], y[test_idx]
#     model = XGBClassifier(n_jobs=-1)

#     model.fit(x_train, y_train, eval_metric='mlogloss')

#     acc_ = model.score(x_test, y_test)
#     acc.append(acc_)
#     print('Acc :', acc_)

# print('Final Acc :', np.mean(acc))
'''
'''
model = Sequential()
model.add(Dense(32, activation='relu', input_shape=(11,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, epochs=300, batch_size=16, validation_split=0.2, verbose=2)

acc = model.evaluate(x_test, y_test)[1]
print('Acc :', acc)
'''