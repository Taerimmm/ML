# m31로 만든 0.95 이상의 n_component=? 를 사용하여 XGB 모델을 만들 것
# GridSearchCV, RandomizedSearchCV

import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier

# 1. 데이터
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x = np.append(x_train, x_test, axis=0)
y = np.append(y_train, y_test, axis=0)

x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])

pca = PCA()
x = pca.fit_transform(x)

cumsum = np.cumsum(pca.explained_variance_ratio_)

d = np.argmax(cumsum >= 0.95) + 1
print('축소된 차원 수 :', d)
# 축소된 차원 수 : 154

pca = PCA(n_components=d)
x = pca.fit_transform(x)

kfold = KFold(n_splits=5, shuffle=True)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/7, random_state=45)

parameters = [
    {"n_estimators":[100,200,300], "learning_rate":[0.001,0.01,0.1,0.3], "max_depth":[4,5,6]},
    {"n_estimators":[90,100,110], "learning_rate":[0.001,0.01,0.1], "max_depth":[4,5,6], "colsample_bytree":[0.6,0.9,1]},
    {"n_estimators":[90,110], "learning_rate":[0.001,0.1,0.5], "max_depth":[4,5,6], "colsample_bytree":[0.6,0.9,1], "colsample_bylevel":[0.6,0.7,0.9]},    
]

CV = [GridSearchCV, RandomizedSearchCV]
result = []
for search in CV:
    model = search(XGBClassifier(n_jobs=-1, use_label_encoder=False), parameters, cv=kfold)

    model.fit(x_train, y_train, eval_metric='mlogloss', verbose=True, eval_set=[(x_train, y_train), (x_test, y_test)],
              early_stopping_rounds=10)
    # eval_metric -> compile에서 metrics랑 같은 역할

    model.save_model('../data/xgb_save/m34_1_{}.xgb.model'.format(search))

    y_pred = model.predict(x_test)
    
    result.append(model.best_estimator_)
    result.append(accuracy_score(y_test, y_pred))
    result.append(model.score(x_test, y_test))

for i, j in enumerate(CV):
    print(j.__name__ + '의 최적의 param :', result[3*i+0])
    print(j.__name__ + '의 최종 정답률 :', result[3*i+1])
    print(j.__name__ + '의 최종 정답률 :', result[3*i+2])

print('=============== model load ===============')

for i in CV:
    model = XGBClassifier()
    model.load_model('../data/xgb_save/m34_1_{}.xgb.model'.format(i))
    print(i.__name__ + '의 최종 정답률 :', model.score(x_test, y_test))