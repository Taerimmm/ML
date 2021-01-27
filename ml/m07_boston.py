import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

# from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# 1. 데이터
dataset = load_boston()
x = dataset.data
y = dataset.target

print(x.shape)  # (506, 13)
print(y.shape)  # (506,)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

for j in [MinMaxScaler, StandardScaler]:
    scaler = j()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    print(j.__name__)
    # 2. 모델
    for i in [LinearRegression, KNeighborsRegressor, DecisionTreeRegressor, RandomForestRegressor]:
        print()
        model = i()

        # 훈련
        model.fit(x_train,y_train)

        y_pred = model.predict(x_test)
        # print('y_test :', y_test)
        # print('y_pred :', y_pred)

        result = model.score(x_test,y_test)
        print(i.__name__ + '\'s score(r2) :', result)
        
        r2 = r2_score(y_test, y_pred)
        print(i.__name__ + '\'s r2_score :', r2)

    if j == StandardScaler:
        break
    print('=================================================================')

'''
MinMaxScaler

LinearRegression's score(r2) : 0.7848279551648666
LinearRegression's r2_score : 0.7848279551648666

KNeighborsRegressor's score(r2) : 0.7602442313449473
KNeighborsRegressor's r2_score : 0.7602442313449473

DecisionTreeRegressor's score(r2) : 0.8621105218120777
DecisionTreeRegressor's r2_score : 0.8621105218120777

RandomForestRegressor's score(r2) : 0.9299825066833465
RandomForestRegressor's r2_score : 0.9299825066833465
=================================================================
StandardScaler

LinearRegression's score(r2) : 0.7848279551648661
LinearRegression's r2_score : 0.7848279551648661

KNeighborsRegressor's score(r2) : 0.8158642498804515
KNeighborsRegressor's r2_score : 0.8158642498804515

DecisionTreeRegressor's score(r2) : 0.8581437147312935
DecisionTreeRegressor's r2_score : 0.8581437147312935

RandomForestRegressor's score(r2) : 0.9270156844118358
RandomForestRegressor's r2_score : 0.9270156844118358
'''

'''
Tensorflow's R2 : 0.9322565193410672
'''