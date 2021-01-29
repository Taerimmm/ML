import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

# 1. 데이터
values = '0.07096581 0.00977218 0.26210786 0.07852836 0.04767144 0.0714857\
 0.04529296 0.02261024 0.3172664  0.07429906'

feature_importances = pd.DataFrame(list(map(float, values.split()))).sort_values(by=0)

percent = 0.25

index = sorted(feature_importances.iloc[int(np.round(len(feature_importances.index)*percent)):,:].index)

print(index)

dataset = load_diabetes()
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
[0.07096581 0.00977218 0.26210786 0.07852836 0.04767144 0.0714857
 0.04529296 0.02261024 0.3172664  0.07429906]

acc : 0.4181988259572217
'''

'''
# 25% 미만 인 feature 제거 후 
[0.07839196 0.26737759 0.08222659 0.04994275 0.06568497 0.05899164
 0.32411604 0.07326847]
 
acc : 0.3881469848676862
'''
