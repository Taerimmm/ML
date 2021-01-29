import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# 1. 데이터
values = '3.00427759e-02 1.53808812e-04 2.66646072e-03 1.13227503e-03\
 3.07877293e-02 3.79559378e-01 8.50877684e-03 9.79148714e-02\
 7.55640378e-04 1.18182875e-02 3.47822800e-02 5.52293760e-03\
 3.96354779e-01'

feature_importances = pd.DataFrame(list(map(float, values.split()))).sort_values(by=0)

percent = 0.25

index = sorted(feature_importances.iloc[int(np.round(len(feature_importances.index)*percent)):,:].index)

print(index)

dataset = load_boston()
x = pd.DataFrame(dataset.data).iloc[:,index]
y = pd.DataFrame(dataset.target)

print(x)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=44)

# 2. 모델
model = GradientBoostingRegressor()

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
[3.00427759e-02 1.53808812e-04 2.66646072e-03 1.13227503e-03
 3.07877293e-02 3.79559378e-01 8.50877684e-03 9.79148714e-02
 7.55640378e-04 1.18182875e-02 3.47822800e-02 5.52293760e-03
 3.96354779e-01]

acc : 0.8941528927104803
'''

'''
# 25% 미만 인 feature 제거 후 
[0.02836551 0.00293281 0.03479936 0.37954049 0.0088569  0.09786892
 0.01213989 0.03356735 0.00570301 0.39622577]
 
acc : 0.8938502491234904
'''