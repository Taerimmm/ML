import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 1. 데이터
values = '0.         0.         0.43843499 0.56156501'

feature_importances = pd.DataFrame(list(map(float, values.split())))

print(list(feature_importances[feature_importances != 0].dropna().index))

dataset = load_iris()
x = pd.DataFrame(dataset.data).iloc[:,list(feature_importances[feature_importances != 0].dropna().index)]
y = pd.DataFrame(dataset.target)

print(x)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=44)

# 2. 모델
model = DecisionTreeClassifier(max_depth=4)

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
    plt.yticks(np.arange(n_features), [dataset.feature_names[x] for x in list(feature_importances[feature_importances != 0].dropna().index)])
    plt.xlabel('Feature Importances')
    plt.ylabel('Features')
    plt.ylim(-1, n_features)

plot_feature_importances_dataset(model)
plt.show()

'''
# model.feature_importances_
[0.         0.         0.43843499 0.56156501]

acc : 0.9
'''

'''
# 0 인 feature 제거 후 
[0.96990618 0.03009382]

acc : 0.9666666666666667
'''