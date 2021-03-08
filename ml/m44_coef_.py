x = [-3, 31, -11, 4, 0, 22, -2, -5, -25, -14]
y = [-3, 65, -19, 11, 3, 47, -1, -7, -47, -25]

print(x, "\n", y)

import matplotlib.pyplot as plt
plt.plot(x,y)
# plt.show()

import pandas as pd
df = pd.DataFrame({'X' : x, 'Y' : y})
print(df)
print(df.shape)

x_train = df.loc[:, 'X']
y_train = df.loc[:, 'Y']
print(x_train.shape, y_train.shape) # (10,) (10,)
print(type(x_train))    # <class 'pandas.core.series.Series'>

x_train = x_train.values.reshape(len(x_train), 1)
print(x_train.shape, y_train.shape) # (10, 1) (10,)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)

score = model.score(x_train, y_train)
print('score :', score)

print('기울기 :', model.coef_)
print('절편 :', model.intercept_)