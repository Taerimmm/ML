# RF로 모델링 하시오!!!

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

datasets = load_boston()
x = datasets.data
y = datasets.target
print(x.shape, y.shape)     # (506, 13) (506,)

pca = PCA()
pca.fit(x)

cumsum = np.cumsum(pca.explained_variance_ratio_)
print('cumsum :', cumsum)

d = np.argmax(cumsum >= 0.9999) + 1
print('cumsum >= 0.9999 :', cumsum >= 0.9999)
print('d :', d)

'''
plt.plot(cumsum)
plt.grid()
plt.show()
'''

pca = PCA(n_components=d)

x = pca.fit_transform(x)
print(x.shape)             # (506, 9)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=45)

kfold = KFold(n_splits=5, shuffle=True)

'''
parameters = [{'n_estimators':[100,200,300],
              'max_depth':[6,8,10],
              'min_samples_split':[2,4,6,8],
              'min_samples_leaf':[1,3,5,7],
              'n_jobs':[-1]}]
parameters = [
    {"n_estimators":[100,200,300], "learning_rate":[0.001,0.01,0.1,0.3], "max_depth":[4,5,6]},
    {"n_estimators":[90,100,110], "learning_rate":[0.001,0.01,0.1], "max_depth":[4,5,6], "colsample_bytree":[0.6,0.9,1]},
    {"n_estimators":[90,110], "learning_rate":[0.001,0.1,0.5], "max_depth":[4,5,6], "colsample_bytree":[0.6,0.9,1], "colsample_bylevel":[0.6,0.7,0.9]},    
]
'''

parameters = [[{'n_estimators':[100,200,300],
              'max_depth':[6,8,10],
              'min_samples_split':[2,4,6,8],
              'min_samples_leaf':[1,3,5,7],
              'n_jobs':[-1]}],
              [
              {"n_estimators":[100,200,300], "learning_rate":[0.001,0.01,0.1,0.3], "max_depth":[4,5,6]},
              {"n_estimators":[90,100,110], "learning_rate":[0.001,0.01,0.1], "max_depth":[4,5,6], "colsample_bytree":[0.6,0.9,1]},
              {"n_estimators":[90,110], "learning_rate":[0.001,0.1,0.5], "max_depth":[4,5,6], "colsample_bytree":[0.6,0.9,1], "colsample_bylevel":[0.6,0.7,0.9]},    
]
]

for i, j in enumerate([RandomForestRegressor, XGBRegressor]):
    if i == 0:
       model = RandomizedSearchCV(j(), parameters[i], cv=kfold)
       model.fit(x_train, y_train)
    else:
       model = RandomizedSearchCV(j(n_jobs=-1, use_label_encoder=False), parameters[i], cv=kfold)
       model.fit(x_train, y_train, eval_metric='logloss')
        
    y_pred = model.predict(x_test)
    print(j.__name__ + '의 최종 R2 :', r2_score(y_test, y_pred))
    print(j.__name__ + '의 최종 R2 :', model.score(x_test, y_test))

# RandomForestRegressor의 최종 R2 : 0.7929724510970513
# XGBRegressor의 최종 R2 : 0.7948564515098473
