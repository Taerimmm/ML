import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

import warnings
warnings.filterwarnings('ignore')

dataset = load_iris()

x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2, random_state=45)

kfold = KFold(n_splits=5, shuffle=True)

parameters = [
    {"n_estimators":[100,200,300], "learning_rate":[0.001,0.01,0.1,0.3], "max_depth":[4,5,6]},
    {"n_estimators":[90,100,110], "learning_rate":[0.001,0.01,0.1], "max_depth":[4,5,6], "colsample_bytree":[0.6,0.9,1]},
    {"n_estimators":[90,110], "learning_rate":[0.001,0.1,0.5], "max_depth":[4,5,6], "colsample_bytree":[0.6,0.9,1], "colsample_bylevel":[0.6,0.7,0.9]},    
]

model = RandomizedSearchCV(XGBClassifier(eval_metric='mlogloss'), parameters, cv=kfold)

model.fit(x_train, y_train)

print('최적의 매개변수 :', model.best_estimator_)

y_pred = model.predict(x_test)
print('최종 정답률 :', accuracy_score(y_test, y_pred))
print('최종 정답률 :', model.score(x_test, y_test))

'''
최적의 매개변수 : XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=0.6,
              colsample_bynode=1, colsample_bytree=0.6, eval_metric='mlogloss',
              gamma=0, gpu_id=-1, importance_type='gain',
              interaction_constraints='', learning_rate=0.1, max_delta_step=0,
              max_depth=5, min_child_weight=1, missing=nan,
              monotone_constraints='()', n_estimators=110, n_jobs=8,
              num_parallel_tree=1, objective='multi:softprob', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=None, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)
최종 정답률 : 0.9666666666666667
'''