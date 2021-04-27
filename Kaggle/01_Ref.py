import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import accuracy_score

train_data = pd.read_csv('./Kaggle/data/train.csv', index_col=0)
test_data = pd.read_csv('./Kaggle/data/test.csv', index_col=0)

target = train_data.pop('Survived')

# print(train_data)
# print(test_data)
# print(target)

# Preprocessing
train_data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
test_data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

train_data['Age'].fillna((train_data['Age'].median()), inplace=True)
test_data['Age'].fillna((train_data['Age'].median()), inplace=True)

train_data['Fare'].fillna((train_data['Fare'].median()), inplace=True)
test_data['Fare'].fillna((train_data['Fare'].median()), inplace=True)

train_data['Fare'] = train_data['Fare'].map(lambda i: np.log(i) if i > 0 else 0)
test_data['Fare'] = test_data['Fare'].map(lambda i: np.log(i) if i > 0 else 0)

train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1

train_data['Embarked'].fillna('S', inplace=True)
test_data['Embarked'].fillna('S', inplace=True)

for col in ['Pclass', 'Sex', 'Embarked']:
    le = LabelEncoder()
    le.fit(train_data[col])
    train_data[col] = le.transform(train_data[col])    
    test_data[col] = le.transform(test_data[col])


x_train, x_valid, y_train, y_valid = train_test_split(train_data, target, test_size=0.1, random_state=0)

x_train_scaled = x_train.copy()
x_valid_scaled = x_valid.copy()
test_scaled = test_data.copy()

scaler = StandardScaler()
scaler.fit(train_data)
x_train_scaled = scaler.transform(x_train)
x_valid_scaled = scaler.transform(x_valid)
test_scaled = scaler.transform(test_data)

x_train_scaled = pd.DataFrame(x_train_scaled, columns=x_train.columns)
x_valid_scaled = pd.DataFrame(x_valid_scaled, columns=x_valid.columns)
test_scaled = pd.DataFrame(test_scaled, columns=test_data.columns)

# SVM with RBF kernel
svc_kernel_rbf = SVC(kernel='rbf', random_state=0, C=0.01, probability=True)
svc_kernel_rbf.fit(x_train_scaled, y_train)
y_pred = svc_kernel_rbf.predict(x_valid_scaled)
print("Accuracy: {}".format(accuracy_score(y_pred, y_valid)))

svc_kernel_rbf_final_pred_probs = svc_kernel_rbf.predict_proba(test_scaled)[:,1]
svc_kernel_rbf_final_pred_binary = svc_kernel_rbf.predict(test_scaled)

submission = pd.read_csv('./Kaggle/data/sample_submission.csv', index_col='PassengerId')
submission['Survived'] = svc_kernel_rbf_final_pred_binary
submission.to_csv('svm_kernel_rbf.csv')

# Logistic Regression¶
log_reg = LogisticRegression(random_state=0)
log_reg.fit(x_train_scaled, y_train)
y_pred = log_reg.predict(x_valid_scaled)
print("Accuracy: {}".format(accuracy_score(y_pred, y_valid)))

log_reg_final_pred_probs = log_reg.predict_proba(test_scaled)[:,1]
log_reg_final_pred_binary = log_reg.predict(test_scaled)
submission['Survived'] = log_reg_final_pred_binary
submission.to_csv('logistic_regression.csv')