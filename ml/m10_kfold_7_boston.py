import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, r2_score

# from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# 1. 데이터
dataset = load_boston()
x = dataset.data
y = dataset.target

print(x.shape, y.shape)  # (506, 13), (506,)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, random_state=45)
kfold = KFold(n_splits=5, shuffle=True)

# 2. 모델
for i in [LinearRegression, KNeighborsRegressor, DecisionTreeRegressor, RandomForestRegressor]:
    print()
    model = i()

    # 훈련
    scores = cross_val_score(model, x_train, y_train, cv=kfold)
    print(i.__name__ + '\'s score(R2) :', scores)

'''
LinearRegression's score(R2)      : [0.68548713 0.73717088 0.7014253  0.60038314 0.78640173]

KNeighborsRegressor's score(R2)   : [0.53018174 0.49457994 0.51838518 0.4337409  0.42476561]

DecisionTreeRegressor's score(R2) : [0.74241591 0.72411056 0.82158985 0.75469256 0.73659862]

RandomForestRegressor's score(R2) : [0.8863208  0.76628895 0.91202741 0.87049145 0.78873223]
'''