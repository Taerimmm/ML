# 모델 : RandomForestClassifier
import numpy as np
from time import time
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
dataset = load_iris()
x = dataset.data
y = dataset.target

print(x.shape, y.shape)  # (150, 4) ,(150,)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=45)

kfold = KFold(n_splits=5, shuffle=True)

parameters = [{'n_estimators':[100, 150, 200, 250]},
              {'max_depth':[6,8,10,12]},
              {'min_samples_leaf':[1,3,5,7,10]},
              {'min_samples_split':[2,3,5,10]},
              {'n_jobs':[-1]}]

# 2. 모델 
model = GridSearchCV(RandomForestClassifier(), parameters, cv=kfold)

score = cross_val_score(model, x_train, y_train, cv=kfold)

print('교차검증점수 :', score)


'''
start = time()
model.fit(x_train, y_train)

print('GridSearchCV took %.2f seconds' % (time() - start))
print('최적의 매개변수 :', model.best_estimator_)

y_pred = model.predict(x_test)
print('최종 정답률 :', accuracy_score(y_test, y_pred))
print('최종 정답률 :', model.score(x_test, y_test))
'''