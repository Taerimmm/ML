import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from sklearn.model_selection import train_test_split
# from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.model_selection import RandomizedSearchCV

from keras.wrappers.scikit_learn import KerasClassifier

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout


# Data
train = pd.read_csv('./Kaggle/data/train.csv')
test = pd.read_csv('./Kaggle/data/test.csv')

pseudo_label = pd.read_csv('./Kaggle/data/pseudo_label.csv', index_col=0)

print(train.shape)
print(test.shape)
print(pseudo_label.shape)

# Feature Engineering
test['Survived'] = [x for x in pseudo_label.Survived]
train['Survived'].value_counts()
train.isnull().sum()
test.isnull().sum()

train['Ticket'].nunique()
train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train[["Embarked", "Survived"]].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_copy = train.copy()
test_copy = test.copy()
train = train.drop(['PassengerId', 'Ticket'], axis = 1)
test = test.drop(['PassengerId', 'Ticket',], axis = 1)
combine = [train, test]

train['Cabin'].fillna('U', inplace=True)
train['Cabin'] = train['Cabin'].apply(lambda x: x[0])

test['Cabin'].fillna('U', inplace=True)
test['Cabin'] = test['Cabin'].apply(lambda x: x[0])

train['Cabin'].unique()
for dataset in combine:
  dataset['Cabin'] = dataset['Cabin'].fillna('U')
  dataset['Cabin'] = dataset['Cabin'].apply(lambda x: x[0])
  
pd.crosstab(train['Cabin'], train['Survived'])
train[['Cabin', 'Survived']].groupby(['Cabin'], as_index = False).mean().sort_values(by = 'Survived', ascending = True)
cabin_mapping = {"T": 0, "U": 1, "A": 2, "G": 3, "C": 4, "F": 5, "B": 6, "E": 7, "D": 8}
for dataset in combine:
    dataset['Cabin'] = dataset['Cabin'].map(cabin_mapping)
    dataset['Cabin'] = dataset['Cabin'].fillna(0)

for dataset in combine:
    dataset['Title'] = dataset['Name'].map(lambda x: x.split(',')[1].split('.')[0].strip())

pd.crosstab(train['Title'], train['Sex'])

for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train['Age'].fillna(train['Age'].dropna().median(), inplace=True)
test['Age'].fillna(train['Age'].mean(), inplace = True)
test['Fare'].fillna(train['Fare'].dropna().median(), inplace = True)
train['Embarked'].fillna('C', inplace = True)
test['Embarked'].fillna('C', inplace = True)
train['Fare'].fillna(train['Fare'].dropna().median(), inplace = True)
train.drop('Title',axis=1,inplace=True)
test.drop('Title',axis=1,inplace=True)

train.isnull().sum()
test.isnull().sum()

train['AgeBand'] = pd.cut(train['Age'], 5)
train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)

for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 4
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age'] = 0

for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train['FareBand'] = pd.qcut(train['Fare'], 4)
train[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)

for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train = train.drop(['FareBand'], axis=1)
#combine = [train, test]
data = pd.concat([train, test], axis=0)


data = pd.concat([train, test], axis=0)
train = data.iloc[:train.shape[0]]
test = data.iloc[train.shape[0]:].drop(columns=['Survived'])


lab_cols = ['Pclass','Age', 'Ticket', 'Fare', 'Cabin', 'Embarked']
target = 'Survived'

features_selected = ['Pclass', 'Sex', 'Age','Embarked','Parch','SibSp','Fare','Cabin']

X = data.drop(target, axis=1)
X = X[features_selected]
y = data[target]

test = test[features_selected]


# Building Model
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=101, test_size=0.25)

def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units=10,kernel_initializer='uniform',activation='relu',input_dim=8))
    classifier.add(Dropout(rate = 0.2))
    classifier.add(Dense(units=64,kernel_initializer='uniform',activation='relu'))
    classifier.add(Dropout(rate = 0.2))
    # classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu'))
    # classifier.add(Dropout(rate = 0.2))
    classifier.add(Dense(units=1,kernel_initializer='uniform',activation='sigmoid'))
    classifier.compile(optimizer=optimizer,loss='binary_crossentropy',metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)
param_grid = dict(optimizer = ['Adam'],
                  epochs=[50,100,200,250],
                  batch_size=[16,32,64,128])
grid = RandomizedSearchCV(classifier, param_grid, scoring='accuracy',cv=10,n_jobs=-1,random_state=101)
grid_result = grid.fit(X_train, y_train)
best_parameters = grid.best_params_
best_accuracy = grid.best_score_