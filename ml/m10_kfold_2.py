import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression # 분류
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# 1. 데이터
dataset = load_iris()
x = dataset.data
y = dataset.target

print(x.shape, y.shape)  # (150, 4) ,(150,)

kfold = KFold(n_splits=5, shuffle=True)

# 2. 모델
model = LinearSVC()

scores = cross_val_score(model, x, y, cv=kfold) # fit, score가 다 포함
print('scores :', scores)

'''
# 훈련
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
# print('y_test :', y_test)
# print('y_pred :', y_pred)

result = model.score(x_test,y_test)
print(i.__name__ + '\'s score(acc) :', result)

acc = accuracy_score(y_test, y_pred)
print(i.__name__ + '\'s accuracy_score :', acc)

'''
