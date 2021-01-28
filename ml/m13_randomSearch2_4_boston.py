import numpy as np
from time import time
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, KFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import accuracy_score, r2_score

from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
dataset = load_boston()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=45)

kfold = KFold(n_splits=5, shuffle=True)

parameters = [{'n_estimators':[100, 150, 200, 250],
              'max_depth':[6,8,10,12],
              'min_samples_leaf':[1,3,5,7,10],
              'min_samples_split':[2,3,5,10],
              'n_jobs':[-1,2,4]}]

# 2. 모델
model = RandomizedSearchCV(RandomForestRegressor(), parameters, cv=kfold)

start = time()
model.fit(x_train,y_train)

print('RandomizedSearchCV took %.2f seconds' % (time() - start))
print('최적의 매개변수 :', model.best_estimator_)

y_pred = model.predict(x_test)
print('최종 R2 :', r2_score(y_test, y_pred))
print('최종 R2 :', model.score(x_test, y_test))

'''
RandomizedSearchCV took 20.57 seconds
최적의 매개변수 : RandomForestRegressor(max_depth=10, n_estimators=150, n_jobs=2)
최종 R2 : 0.9061699915768333
최종 R2 : 0.9061699915768333
'''