# 데이터는 RandomForest 사용
# pipeline 엮어서 25번 돌리기!!!
# 데이터는 wine

import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.ensemble import RandomForestClassifier

# 1. 데이터
dataset = load_wine()
x = dataset.data
y = dataset.target

kfold = KFold(n_splits=5, shuffle=True)

parameters = [{'randomforestclassifier__n_estimators':[100,200,300],
              'randomforestclassifier__max_depth':[6,8,10],
              'randomforestclassifier__min_samples_split':[2,4,6,8],
              'randomforestclassifier__min_samples_leaf':[1,3,5,7],
              'randomforestclassifier__n_jobs':[-1]}]

# 2. 모델
for scaler in [MinMaxScaler, StandardScaler]:
    pipe = make_pipeline(MinMaxScaler(), RandomForestClassifier())
    model = RandomizedSearchCV(pipe, parameters, cv=kfold)

    scores = cross_val_score(model, x, y, cv=kfold)

    print('교차검증점수 :', scores)
