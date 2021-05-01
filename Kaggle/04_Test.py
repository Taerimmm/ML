import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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

all_df = pd.concat([train_data, test_data])

print(all_df.head())
print(all_df.tail())

target2 = pd.read_csv('./Kaggle/data/pseudo_label.csv')['Survived']

target = pd.concat([target, target2], ignore_index=True)

submission = pd.read_csv('./Kaggle/data/sample_submission.csv', index_col='PassengerId')


# print(all_df.shape)
# print(test_data.shape)
# print(target)

# Preprocessing
all_df.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
test_data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

all_df['Age'].fillna((all_df['Age'].median()), inplace=True)
test_data['Age'].fillna((test_data['Age'].median()), inplace=True)

all_df['Fare'].fillna((all_df['Fare'].median()), inplace=True)
test_data['Fare'].fillna((test_data['Fare'].median()), inplace=True)

all_df['Fare'] = all_df['Fare'].map(lambda i: np.log(i) if i > 0 else 0)
test_data['Fare'] = test_data['Fare'].map(lambda i: np.log(i) if i > 0 else 0)

all_df['FamilySize'] = all_df['SibSp'] + all_df['Parch'] + 1
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1

all_df['Embarked'].fillna('S', inplace=True)
test_data['Embarked'].fillna('S', inplace=True)

for col in ['Pclass', 'Sex', 'Embarked']:
    le = LabelEncoder()
    le.fit(all_df[col])
    all_df[col] = le.transform(all_df[col])    
    test_data[col] = le.transform(test_data[col])



x_train, x_valid, y_train, y_valid = train_test_split(all_df, target, test_size=0.1, random_state=0)

x_train_scaled = x_train.copy()
x_valid_scaled = x_valid.copy()
test_scaled = test_data.copy()

scaler = StandardScaler()
scaler.fit(all_df)
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
submission['Survived'] = svc_kernel_rbf_final_pred_binary
submission.to_csv('./Kaggle/data/svm_kernel_rbf.csv')

# Logistic Regression
log_reg = LogisticRegression(random_state=0)
log_reg.fit(x_train_scaled, y_train)
y_pred = log_reg.predict(x_valid_scaled)
print("Accuracy: {}".format(accuracy_score(y_pred, y_valid)))

log_reg_final_pred_probs = log_reg.predict_proba(test_scaled)[:,1]
log_reg_final_pred_binary = log_reg.predict(test_scaled)
submission['Survived'] = log_reg_final_pred_binary
submission.to_csv('./Kaggle/data/logistic_regression.csv')

# Random Forest
random_forest = RandomForestClassifier(random_state=0, n_estimators=1000, max_features=2, min_samples_split=0.1)
random_forest.fit(x_train, y_train)
y_pred = random_forest.predict(x_valid)
print("Accuracy: {}".format(accuracy_score(y_pred, y_valid)))

random_forest_final_pred_probs = random_forest.predict_proba(test_data)[:,1]
random_forest_final_pred_binary = random_forest.predict(test_data)
submission['Survived'] = random_forest_final_pred_binary
submission.to_csv('./Kaggle/data/random_forest.csv')

# XGBoost
xgboost = XGBClassifier(random_state=0, n_estimators=1000, use_label_encoder=False, eval_metric='logloss')
xgboost.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], early_stopping_rounds=10, verbose=False)
y_pred = xgboost.predict(x_valid)
print("Accuracy: {}".format(accuracy_score(y_pred, y_valid)))

xgboost_final_pred_probs = xgboost.predict_proba(test_data)[:,1]
xgboost_final_pred_binary = xgboost.predict(test_data)
submission['Survived'] = xgboost_final_pred_binary
submission.to_csv('./Kaggle/data/xgboost.csv')

# LGBM
lgbm = LGBMClassifier(random_state=0, n_estimators=1000)
lgbm.fit(x_train, y_train, eval_set=(x_valid, y_valid), eval_metric='logloss', early_stopping_rounds=10, verbose=0)
y_pred = lgbm.predict(x_valid)
print("Accuracy: {}".format(accuracy_score(y_pred, y_valid)))

lgbm_final_pred_probs = lgbm.predict_proba(test_data)[:,1]
lgbm_final_pred_binary = lgbm.predict(test_data)
submission['Survived'] = lgbm_final_pred_binary
submission.to_csv('./Kaggle/data/lgbm.csv')

# CatBoost
catboost = CatBoostClassifier(random_state=0, n_estimators=1000)
catboost.fit(x_train, y_train, eval_set=(x_valid, y_valid), verbose=False, early_stopping_rounds=10)
y_pred = catboost.predict(x_valid)
print("Accuracy: {}".format(accuracy_score(y_pred, y_valid)))

catboost_final_pred_probs = catboost.predict_proba(test_data)[:,1]
catboost_final_pred_binary = catboost.predict(test_data)
submission['Survived'] = catboost_final_pred_binary
submission.to_csv('./Kaggle/data/catboost.csv')

# Neural Network
tf.random.set_seed(0)

early_stopping = keras.callbacks.EarlyStopping(patience=10, min_delta=0.001, restore_best_weights=True)

neural_net = keras.Sequential([
    layers.Dense(100, activation='relu', input_shape=[x_train_scaled.shape[1]]),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(100, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(50, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(1, activation='sigmoid')
])

neural_net.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])

history = neural_net.fit(x_train, y_train, validation_data=(x_valid, y_valid), batch_size=512, epochs=50, callbacks=[early_stopping])


neural_net_final_pred_probs = neural_net.predict(test_data).reshape(100000,)
neural_net_final_pred_binary = np.round(neural_net_final_pred_probs).astype(int).reshape(100000,)
submission['Survived'] = neural_net_final_pred_binary
submission.to_csv('./Kaggle/data/neural_net.csv')


# Hard voting classifier
binary_average = np.mean([svc_kernel_rbf_final_pred_binary,
                          log_reg_final_pred_binary,
                          random_forest_final_pred_binary,
                          xgboost_final_pred_binary,
                          lgbm_final_pred_binary,
                          catboost_final_pred_binary,
                          neural_net_final_pred_binary], axis=0)

hard_classifier_predictions = np.round(binary_average).astype(int)

submission['Survived'] = hard_classifier_predictions
submission.to_csv('./Kaggle/data/hard_voting_classifier.csv')

# 0.80976

# Soft voting classifier
probs_average = np.mean([svc_kernel_rbf_final_pred_probs,
                         log_reg_final_pred_probs,
                         random_forest_final_pred_probs,
                         xgboost_final_pred_probs,
                         lgbm_final_pred_probs,
                         catboost_final_pred_probs,
                         neural_net_final_pred_probs], axis=0)

soft_classifier_predictions = np.round(probs_average).astype(int)

submission['Survived'] = soft_classifier_predictions
submission.to_csv('./Kaggle/data/soft_voting_classifier.csv')

# 0.80923