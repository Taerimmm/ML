import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC, SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression # 분류
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier

import warnings

warnings.filterwarnings('ignore')

# 1. 데이터
dataset = load_iris()
x = dataset.data
y = dataset.target

print(x.shape, y.shape)  # (150, 4) ,(150,)

dataset = pd.read_csv('../data/csv/iris_sklearn.csv', header=0, index_col=0)
x = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

print(x.shape, y.shape)  # (150, 4) ,(150,)

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=77, shuffle=True)

kfold = KFold(n_splits=5, shuffle=True)

parameters = [{"C":[1, 10, 100, 1000], "kernel":['linear']},
              {"C":[1, 10, 100], 'kernel':['rbf'], 'gamma':[0.001, 0.0001]},
              {"C":[1, 10, 100, 1000], 'kernel':['sigmoid'], 'gamma':[0.001, 0.0001]}]

# 2. 모델
model = RandomizedSearchCV(SVC(), parameters, cv=kfold)

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가 예측
print('최적의 매개변수 :', model.best_estimator_) # 총 90번 훈련 (4 + 3 * 2 + 4 * 2) * 5

y_pred = model.predict(x_test)
print('최종 정답률 :', accuracy_score(y_test, y_pred))

score = model.score(x_test, y_test)
print(score)