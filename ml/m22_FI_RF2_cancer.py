import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# 1. 데이터
values = '0.03295345 0.01535239 0.06074372 0.03395368 0.00571023 0.02363396\
 0.04179621 0.10641886 0.00589289 0.00458254 0.01038342 0.00480702\
 0.00874506 0.04547727 0.0038623  0.00438753 0.00449128 0.00547384\
 0.00318735 0.00399546 0.11563954 0.02161321 0.15021738 0.11807957\
 0.01886413 0.01932773 0.03723519 0.07915724 0.00497594 0.0090416'

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
[0.03295345 0.01535239 0.06074372 0.03395368 0.00571023 0.02363396
 0.04179621 0.10641886 0.00589289 0.00458254 0.01038342 0.00480702
 0.00874506 0.04547727 0.0038623  0.00438753 0.00449128 0.00547384
 0.00318735 0.00399546 0.11563954 0.02161321 0.15021738 0.11807957
 0.01886413 0.01932773 0.03723519 0.07915724 0.00497594 0.0090416 ]

acc : 0.9649122807017544
'''

'''
# 25% 미만 인 feature 제거 후 
[0.02632533 0.01717122 0.05391192 0.05437326 0.00903047 0.02236841
 0.07560854 0.1119374  0.00391443 0.01440554 0.00664839 0.02562352
 0.00506054 0.07488295 0.02452176 0.18197658 0.11541469 0.02164526
 0.01032669 0.03314018 0.10110583 0.01060708]
 
acc : 0.9649122807017544
'''