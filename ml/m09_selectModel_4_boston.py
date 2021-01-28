from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.utils import all_estimators
from sklearn.datasets import load_boston
import warnings

warnings.filterwarnings('ignore')

dataset = load_boston()
x = dataset.data
y = dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=44)

import sklearn
print(sklearn.__version__) # 0.23.2

# all_estimators -> 0.20 에 최적화되어있다.

allAlgorithms = all_estimators(type_filter='regressor') # sklearn의 분류형 모델 전체
# print(allAlgorithms)

for (name, algorithm) in allAlgorithms:
    try :
        model = algorithm()

        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(name, '의 R2 :', r2_score(y_test, y_pred))
    except :
        # continue
        print(name, '은 없는 놈') # 0.23.2 에 없는 algorithm

# 기준이 되는 지표로 삼을 수 있다.
'''
# Max-Score Algorithm
ExtraTreesRegressor 의 R2 : 0.905771991917713
'''

'''
Tensorflow's R2 : 0.9322565193410672
'''