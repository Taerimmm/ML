# 실습
# RandomSearch, GS와 Pipeline을 엮어라!!
# 모델은 RandomForest

import numpy as np
from sklearn.datasets import load_wine
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import warnings
warnings.filterwarnings('ignore')

# 1. 데이터
dataset = load_wine()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=45)

kfold = KFold(n_splits=5, shuffle=True)

parameters = [{'RandomForest__n_estimators':[100,200,300]},
              {'RandomForest__max_depth':[6,8,10]},
              {'RandomForest__min_samples_split':[2,4,6,8]},
              {'RandomForest__min_samples_leaf':[1,3,5,7]},
              {'RandomForest__n_jobs':[-1]}]

# parameters = [{'randomforestclassifier__n_estimators':[100,200,300]},
#               {'randomforestclassifier__max_depth':[6,8,10]},
#               {'randomforestclassifier__min_samples_split':[2,4,6,8]},
#               {'randomforestclassifier__min_samples_leaf':[1,3,5,7]},
#               {'randomforestclassifier__n_jobs':[-1]}]

# 2. 모델
for scaler in [MinMaxScaler, StandardScaler]:
    for search in [GridSearchCV, RandomizedSearchCV]:
        pipe = Pipeline([(scaler.__name__, scaler()), ('RandomForest', RandomForestClassifier())])
        # pipe = make_pipeline(scaler(), RandomForestClassifier())

        model = search(pipe, parameters, cv=kfold)

        model.fit(x_train, y_train)

        result = model.score(x_test, y_test)
        
        print('Scaler :', scaler.__name__, '\t SearchCV :', search.__name__)
        print('RandomForestClassifier의 최적의 매개변수 :', model.best_estimator_)
        print('최종 정답률', result)
        print('=========================================')
