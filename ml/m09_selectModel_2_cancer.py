from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import all_estimators
from sklearn.datasets import load_breast_cancer
import warnings

warnings.filterwarnings('ignore')

dataset = load_breast_cancer()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=44)

import sklearn
print(sklearn.__version__) # 0.23.2

# all_estimators -> 0.20 에 최적화되어있다.

allAlgorithms = all_estimators(type_filter='classifier') # sklearn의 분류형 모델 전체
# print(allAlgorithms)

for (name, algorithm) in allAlgorithms:
    try :
        model = algorithm()

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name, '의 정답율 :', accuracy_score(y_test, y_pred))
    except :
        # continue
        print(name, '은 없는 놈') # 0.23.2 에 없는 algorithm

# 기준이 되는 지표로 삼을 수 있다.
'''
# Max-Score Algorithm
LinearDiscriminantAnalysis 의 정답율 : 0.9912280701754386
'''

'''
Tensorflow's acc : 0.9824561476707458
'''