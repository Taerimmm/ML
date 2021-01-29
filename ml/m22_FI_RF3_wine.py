import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

# 1. 데이터
values = '0.14135756 0.0359451  0.01606806 0.0149352  0.02423362 0.05379849\
 0.14459695 0.00749992 0.01658677 0.15177966 0.08089725 0.13235149\
 0.17994993'

feature_importances = pd.DataFrame(list(map(float, values.split()))).sort_values(by=0)

percent = 0.25

index = sorted(feature_importances.iloc[int(np.round(len(feature_importances.index)*percent)):,:].index)

print(index)

dataset = load_wine()
x = pd.DataFrame(dataset.data).iloc[:,index]
y = pd.DataFrame(dataset.target)

print(x)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=44)

# 2. 모델
model = RandomForestClassifier()

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
[0.14135756 0.0359451  0.01606806 0.0149352  0.02423362 0.05379849
 0.14459695 0.00749992 0.01658677 0.15177966 0.08089725 0.13235149
 0.17994993]

acc : 0.9722222222222222
'''

'''
# 25% 미만 인 feature 제거 후 
[0.16446761 0.02885843 0.02148231 0.04491047 0.21528165 0.02237456
 0.14455156 0.0749835  0.12387622 0.15921369]
 
acc : 0.9722222222222222
'''