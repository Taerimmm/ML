# m31로 만든 1.0 이상의 n_component=? 를 사용하여 XGB 모델을 만들 것
# dnn과 비교

import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.decomposition import PCA

from xgboost import XGBClassifier

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x = np.append(x_train, x_test, axis=0)
y = np.append(y_train, y_test, axis=0)

x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])

pca = PCA()
x = pca.fit_transform(x)

cumsum = np.cumsum(pca.explained_variance_ratio_)

d = np.argmax(cumsum >= 1.0) + 1
print('축소된 차원 수 :', d)
# 축소된 차원 수 : 713
                         
pca = PCA(n_components=d)
x = pca.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/7, random_state=45)

kfold = KFold(n_splits=5, shuffle=True)

model = XGBClassifier(n_jobs=8, use_label_encoder=False)

model.fit(x_train, y_train, eval_metric='mlogloss')

score = model.score(x_test, y_test)
print('acc :', score)

# acc : 0.9606
