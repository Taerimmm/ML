import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

import warnings
warnings.filterwarnings('ignore')

X = np.load('./project/mini/data/X.npy')
y = pd.read_csv('./project/mini/data/y_label.csv', header=0).iloc[:,0]

X = X.reshape(X.shape[0], X.shape[1]*X.shape[2])
print(X.shape)  # (6194, 84480)
print(y.shape)  # (6194,)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

kfold = KFold(n_splits=5, shuffle=True)

parameters = [
    {"objective":['multi:softmax'],
     "n_estimators":[100, 200, 400, 800],
     "learning_rate":[0.001, 0.01, 0.1],
     "max_depth":[4,5,6]},    
]

model = RandomizedSearchCV(XGBClassifier(eval_metric='mlogloss'), parameters, cv=kfold)

model.fit(x_train, y_train)

print('최적의 매개변수 :', model.best_estimator_)

y_pred = model.predict(x_test)
print('최종 정답률 :', accuracy_score(y_test, y_pred))
print('최종 정답률 :', model.score(x_test, y_test))

'''
최적의 매개변수 : XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=0.6,
              colsample_bynode=1, colsample_bytree=1, eval_metric='mlogloss',
              gamma=0, gpu_id=-1, importance_type='gain',
              interaction_constraints='', learning_rate=0.1, max_delta_step=0,
              max_depth=5, min_child_weight=1, missing=nan,
              monotone_constraints='()', n_estimators=100, n_jobs=12,
              num_parallel_tree=1, objective='multi:softprob', random_state=0,
              reg_alpha=0, reg_lambda=1, scale_pos_weight=None, subsample=1,
              tree_method='exact', validate_parameters=1, verbosity=None)
최종 정답률 : 0.549636803874092
최종 정답률 : 0.549636803874092
'''