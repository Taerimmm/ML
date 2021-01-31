import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import boston_housing
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline, make_pipeline

from sklearn.svm import LinearSVR, SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings('ignore')


dataset = boston_housing
(x_train, y_train), (x_test, y_test) = dataset.load_data()

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

kfold = KFold(n_splits=5, shuffle=True)

parameters = [{'randomforestregressor__n_estimators':[100,150],
               'randomforestregressor__max_depth':[2,4],
               'randomforestregressor__min_samples_split':[2,4],
               'randomforestregressor__min_samples_leaf':[1,3],
               'randomforestregressor__n_jobs':[-1]}]

pipe = Pipeline([('MinMax',MinMaxScaler()), ('randomforestregressor',RandomForestRegressor())])

model = GridSearchCV(pipe, parameters, cv=kfold)

model.fit(x_train, y_train)

print(model.score(x_test, y_test))

print('best_estimator :', model.best_estimator_)

# pipeline 에서 feature_importances_ ??

'''
def plot_feature_importances_dataset(model):
    n_features = x_train.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), x_train.feature_names)
    plt.xlabel('Feature Importances')
    plt.ylabel('Features')
    plt.ylim(-1, n_features)

plot_feature_importances_dataset(best_estimator)
plt.show()
'''