# 0.5 이상

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from xgboost import XGBRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = XGBRegressor(n_jobs=8)

param = [
    {"n_estimators":[100,200,300], "learning_rate":[0.001,0.01,0.1,0.3], "max_depth":[4,5,6]},
    {"n_estimators":[90,100,110], "learning_rate":[0.001,0.01,0.1], "max_depth":[4,5,6], "colsample_bytree":[0.6,0.9,1]},
    {"n_estimators":[90,110], "learning_rate":[0.001,0.1,0.5], "max_depth":[4,5,6], "colsample_bytree":[0.6,0.9,1], "colsample_bylevel":[0.6,0.7,0.9]},    
]

r2 = []
CV = [GridSearchCV, RandomizedSearchCV]
for search in CV:
    print(search.__name__)
    model_ = search(model, param)

    model_.fit(x_train, y_train)
    
    print('R2 :', model_.score(x_test, y_test))
    r2.append(model_.score(x_test, y_test))

    # print(model_.__dir__())
    print(model_.best_score_)
    print(model_.best_params_)
    print(model_.best_estimator_)

    best_model = model_.best_estimator_

    # print("Feature_importance :\n", model_.feature_importances_)
    
    thresholds = np.sort(model_.best_estimator_.feature_importances_)
    print(thresholds)
     
    print('=======================')
    break

score_list = []
for i, thresh in enumerate(thresholds):
    selection = SelectFromModel(best_model, threshold=thresh, prefit=True)

    select_x_train = selection.transform(x_train)
    print(select_x_train.shape)

    selection_model = XGBRegressor(n_jobs=8)
    selection_model.set_params(**model_.best_params_)
    selection_model.fit(select_x_train, y_train)

    select_x_test = selection.transform(x_test)

    y_predict = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_predict)
    score_list.append(score)

    print('Thresh=%.3f, n=%d, R2:%.2f%%' %(thresh, select_x_train.shape[1], score*100))

    if i == 0:
        reduce_x_train = select_x_train 
        reduce_x_test = select_x_test
        reduce_column_num = select_x_train.shape[1]
        continue

    if (score > max(score_list[:-1])):
        reduce_x_train = select_x_train 
        reduce_x_test = select_x_test
        reduce_column_num = select_x_train.shape[1]

feature_num = len(score_list) - score_list.index(max(score_list))
print('feature 개수 :', feature_num)
# print('feature 개수 :', reduce_column_num)


# 1. PCA
pca = PCA(n_components=feature_num)

decomposition_x_train = pca.fit_transform(x_train)
decomposition_x_test = pca.transform(x_test)
print('PCA :', decomposition_x_train.shape)

model = XGBRegressor(n_jobs=8)

param = [
    {"n_estimators":[100,200,300], "learning_rate":[0.1,0.01], "max_depth":[4,5,6]},
    {"n_estimators":[90,100,110], "learning_rate":[0.1,0.01,0.001], "max_depth":[4,5,6], "colsample_bytree":[0.6,0.9,1]}
]

CV = [GridSearchCV, RandomizedSearchCV]
for search in CV:
    print(search.__name__)
    model_ = search(model, param)

    model_.fit(decomposition_x_train, y_train)
    
    print('R2 :', model_.score(decomposition_x_test, y_test))
    r2.append(model_.score(decomposition_x_test, y_test))

    print('=======================')
    break

# 2. 아예 삭제
model = XGBRegressor(n_jobs=8)

param = [
    {"n_estimators":[100,200,300], "learning_rate":[0.1,0.01], "max_depth":[4,5,6]},
    {"n_estimators":[90,100,110], "learning_rate":[0.1,0.01,0.001], "max_depth":[4,5,6], "colsample_bytree":[0.6,0.9,1]}
]

CV = [GridSearchCV, RandomizedSearchCV]
for search in CV:
    print(search.__name__)
    model_ = search(model, param)

    model_.fit(reduce_x_train, y_train)
    
    print('R2 :', model_.score(reduce_x_test, y_test))
    r2.append(model_.score(reduce_x_test, y_test))

    print('=======================')
    break

print('Before  R2 :', r2[0])
print('AFter 1 R2 :', r2[1])
print('AFter 2 R2 :', r2[2])