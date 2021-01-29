import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# 1. 데이터
values = '6.17759927e-04 9.39672044e-03 2.00656937e-04 1.27330334e-03\
 3.26630055e-04 2.58283802e-03 4.60580631e-04 4.05040738e-01\
 4.76221226e-04 1.66930352e-03 1.60653764e-03 1.82106497e-03\
 1.25356104e-03 7.57684953e-03 3.72634912e-04 6.33242388e-04\
 1.46805369e-03 9.28171533e-04 1.72654168e-03 3.34140672e-03\
 7.12560413e-02 6.09234367e-02 2.86171148e-01 4.53265803e-02\
 1.33036916e-02 1.09853408e-03 2.03596268e-02 5.55056856e-02\
 1.21915803e-03 2.06328154e-03'

feature_importances = pd.DataFrame(list(map(float, values.split()))).sort_values(by=0)

percent = 0.25

index = sorted(feature_importances.iloc[int(np.round(len(feature_importances.index)*percent)):,:].index)

print(index)

dataset = load_breast_cancer()
x = pd.DataFrame(dataset.data).iloc[:,index]
y = pd.DataFrame(dataset.target)

print(x)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=44)

# 2. 모델
model = GradientBoostingClassifier()

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
acc = model.score(x_test, y_test)

print(model.feature_importances_)
print('acc :', acc)

import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importances_dataset(model):
    n_features = x.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features), [dataset.feature_names[x] for x in index])
    plt.xlabel('Feature Importances')
    plt.ylabel('Features')
    plt.ylim(-1, n_features)

plot_feature_importances_dataset(model)
plt.show()

'''
# model.feature_importances_
[6.17759927e-04 9.39672044e-03 2.00656937e-04 1.27330334e-03
 3.26630055e-04 2.58283802e-03 4.60580631e-04 4.05040738e-01
 4.76221226e-04 1.66930352e-03 1.60653764e-03 1.82106497e-03
 1.25356104e-03 7.57684953e-03 3.72634912e-04 6.33242388e-04
 1.46805369e-03 9.28171533e-04 1.72654168e-03 3.34140672e-03
 7.12560413e-02 6.09234367e-02 2.86171148e-01 4.53265803e-02
 1.33036916e-02 1.09853408e-03 2.03596268e-02 5.55056856e-02
 1.21915803e-03 2.06328154e-03]

acc : 0.9736842105263158
'''

'''
# 25% 미만 인 feature 제거 후 
[1.37665386e-02 6.20024372e-04 3.40409626e-03 4.04998239e-01
 1.60457000e-03 3.33584310e-03 2.02778605e-03 1.63564046e-03
 8.40553545e-03 1.24138074e-03 4.88354941e-04 2.75896823e-03
 7.58506403e-02 6.16088666e-02 2.86018940e-01 4.06289161e-02
 1.29226480e-02 4.23438766e-04 1.95126533e-02 5.73848199e-02
 3.31792732e-04 1.03030748e-03]
 
acc : 0.9736842105263158
'''