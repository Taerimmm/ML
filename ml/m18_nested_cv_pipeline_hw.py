# 데이터는 RandomForest 사용
# pipeline 엮어서 25번 돌리기!!!
# 데이터는 diabetes

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.ensemble import RandomForestRegressor

# 1. 데이터
dataset = load_diabetes()
x = dataset.data
y = dataset.target

kfold = KFold(n_splits=5, shuffle=True)

parameters = [{'RandomForest__n_estimators':[100,200,300],
              'RandomForest__max_depth':[6,8,10],
              'RandomForest__min_samples_split':[2,4,6,8],
              'RandomForest__min_samples_leaf':[1,3,5,7],
              'RandomForest__n_jobs':[-1]}]

# 2. 모델
for scaler in [MinMaxScaler, StandardScaler]:
    pipe = Pipeline([(scaler.__name__, scaler()), ('RandomForest', RandomForestRegressor())])
    model = GridSearchCV(pipe, parameters, cv=kfold)

    scores = cross_val_score(model, x, y, cv=kfold)

    print('Scaler :', scaler.__name__)
    print('교차검증점수 :', scores)
