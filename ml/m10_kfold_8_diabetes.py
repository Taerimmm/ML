import numpy as np
from sklearn.datasets import load_diabetes
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
dataset = load_diabetes()
x = dataset.data
y = dataset.target

print(x.shape, y.shape)  # (442, 10) (442,)

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
LinearRegression's score(R2)      : [0.43927814 0.55264136 0.36280019 0.50007606 0.35012461]

KNeighborsRegressor's score(R2)   : [0.47517088 0.46357035 0.41234169 0.30249363 0.23373401]

DecisionTreeRegressor's score(R2) : [-0.41724222  0.20656622 -0.00241595 -0.12564012 -0.04489954]

RandomForestRegressor's score(R2) : [0.3627855  0.37858172 0.37526086 0.48824071 0.47607181]
'''