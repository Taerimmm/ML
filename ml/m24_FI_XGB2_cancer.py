import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline, make_pipeline

# cmd -> pip install xgboost
from xgboost import XGBClassifier

# 1. 데이터
dataset = load_breast_cancer()

x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2, random_state=44)

# n_jobs = -1, 8, 4, 1 속도 비교
for n_jobs in [-1,8,4,1]:
    start = datetime.datetime.now()

    # 2. 모델
    model = XGBClassifier(n_jobs=n_jobs, use_label_encoder=False)

    # 3. 훈련
    model.fit(x_train, y_train, eval_metric='logloss')

    # 4. 평가, 예측
    acc = model.score(x_test, y_test)

    end = datetime.datetime.now()

    # print(model.feature_importances_)
    # print('acc :', acc)
    print('n_jobs =', n_jobs, '일때 걸리는 시간 :', end - start)

'''
def plot_feature_importances_dataset(model):
    n_features = dataset.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), dataset.feature_names)
    plt.xlabel('Feature Importances')
    plt.ylabel('Features')
    plt.ylim(-1, n_features)

plot_feature_importances_dataset(model)
plt.show()
'''