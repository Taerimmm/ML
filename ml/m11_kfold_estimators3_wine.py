from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators
from sklearn.datasets import load_wine
import warnings

warnings.filterwarnings('ignore')

dataset = load_wine()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=44)

kfold = KFold(n_splits=5, shuffle=True)

import sklearn
print(sklearn.__version__) # 0.23.2

# all_estimators -> 0.20 에 최적화되어있다.

allAlgorithms = all_estimators(type_filter='classifier') # sklearn의 분류형 모델 전체
# print(allAlgorithms)

for (name, algorithm) in allAlgorithms:
    try :
        model = algorithm()

        scores = cross_val_score(model, x_train, y_train, cv=kfold) # cv=5 도 가능 / 이때 shuffle=False
        print(name, '의 정답율 :\n', scores)
    except :
        # continue
        print(name, '은 없는 놈') # 0.23.2 에 없는 algorithm

# 기준이 되는 지표로 삼을 수 있다.
