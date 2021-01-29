import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# 1. 데이터
values = '0.04074844 0.00115273 0.00581099 0.00102294 0.02151743 0.38941626\
 0.01676376 0.07772585 0.00481967 0.01222754 0.01851469 0.01105444\
 0.39922527'

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
model = RandomForestRegressor()

# 3. 훈련
model.fit(x_train, y_train)

# 4. 평가, 예측
acc = model.score(x_test, y_test)

print(model.feature_importances_)
print('acc :', acc)


# feature_importances
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

# 25% 미만 인놈 제거

'''
# model.feature_importances_
[0.04074844 0.00115273 0.00581099 0.00102294 0.02151743 0.38941626
 0.01676376 0.07772585 0.00481967 0.01222754 0.01851469 0.01105444
 0.39922527]

acc : 0.8938944683821166
'''

'''
# 25% 미만 인 feature 제거 후 
[0.0359651  0.00586746 0.0233024  0.42482559 0.01487125 0.08197473
 0.01363846 0.01876085 0.01196589 0.36882826]
 
acc : 0.886376802009138
'''