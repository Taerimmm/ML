import pandas as pd
import numpy as np
import seaborn as sns
import itertools
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler

import warnings
warnings.simplefilter('ignore')

from sklearn import model_selection
from sklearn import metrics
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier
from catboost import CatBoostClassifier

from mlxtend.classifier import StackingCVClassifier
import shap

RANDOM_SEED = 2021
PROBAS = True
FOLDS = 10
N_ESTIMATORS = 1000

TARGET = 'Survived'

train_df = pd.read_csv('./Kaggle/data/train.csv')
test_df = pd.read_csv('./Kaggle/data/test.csv')
submission = pd.read_csv('./Kaggle/data/sample_submission.csv')

# Pseudo labels taken from great BIZEN notebook: https://www.kaggle.com/hiro5299834/tps-apr-2021-pseudo-labeling-voting-ensemble
pseudo_labels = pd.read_csv("./Kaggle/data/pseudo_label.csv")
test_df[TARGET] = pseudo_labels[TARGET]

all_df = pd.concat([train_df, test_df]).reset_index(drop=True)

# Age fillna with mean age for each class
all_df['Age'] = all_df['Age'].fillna(all_df['Age'].mean())

# Cabin, fillna with 'X' and take first letter
all_df['Cabin'] = all_df['Cabin'].fillna('X').map(lambda x: x[0].strip())

# Ticket, fillna with 'X', split string and take first split 
all_df['Ticket'] = all_df['Ticket'].fillna('X').map(lambda x:str(x).split()[0] if len(str(x).split()) > 1 else 'X')

# Fare, fillna with mean value
fare_map = all_df[['Fare', 'Pclass']].dropna().groupby('Pclass').median().to_dict()
all_df['Fare'] = all_df['Fare'].fillna(all_df['Pclass'].map(fare_map['Fare']))
all_df['Fare'] = np.log1p(all_df['Fare'])

# Embarked, fillna with 'X' value
all_df['Embarked'] = all_df['Embarked'].fillna('X')

# Name, take only surnames
all_df['Name'] = all_df['Name'].map(lambda x: x.split(',')[0])

label_cols = ['Name', 'Ticket', 'Sex']
onehot_cols = ['Cabin', 'Embarked']
numerical_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']

def label_encoder(c):
    le = LabelEncoder()
    return le.fit_transform(c)

scaler = StandardScaler()

onehot_encoded_df = pd.get_dummies(all_df[onehot_cols])
label_encoded_df = all_df[label_cols].apply(label_encoder)
numerical_df = pd.DataFrame(scaler.fit_transform(all_df[numerical_cols]), columns=numerical_cols)
target_df = all_df[TARGET]

all_df = pd.concat([numerical_df, label_encoded_df, onehot_encoded_df, target_df], axis=1)

lgb_params = {
    'metric': 'binary_logloss',
    'n_estimators': N_ESTIMATORS,
    'objective': 'binary',
    'random_state': RANDOM_SEED,
    'learning_rate': 0.01,
    'min_child_samples': 150,
    'reg_alpha': 3e-5,
    'reg_lambda': 9e-2,
    'num_leaves': 20,
    'max_depth': 16,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'subsample_freq': 2,
    'max_bin': 240,
}

ctb_params = {
    'bootstrap_type': 'Poisson',
    'loss_function': 'Logloss',
    'eval_metric': 'Logloss',
    'random_seed': RANDOM_SEED,
    'task_type': 'GPU',
    'max_depth': 8,
    'learning_rate': 0.01,
    'n_estimators': N_ESTIMATORS,
    'max_bin': 280,
    'min_data_in_leaf': 64,
    'l2_leaf_reg': 0.01,
    'subsample': 0.8
}

rf_params = {
    'max_depth': 15,
    'min_samples_leaf': 8,
    'random_state': RANDOM_SEED
}

# Here you can declare list of classifiers for prototyping purposes

# I do not make any hyperparameter optimization - just taken as they are - here is room for improvement
cl1 = KNeighborsClassifier(n_neighbors = 1)
cl2 = RandomForestClassifier(**rf_params)
cl3 = GaussianNB()
cl4 = DecisionTreeClassifier(max_depth = 5)
cl5 = CatBoostClassifier(**ctb_params, verbose = None, logging_level = 'Silent')
cl6 = LGBMClassifier(**lgb_params)


# I used some hyperparameter search (ExtraTrees - Genetic search)
cl7 = ExtraTreesClassifier(bootstrap=False, criterion='entropy', max_features=0.55, min_samples_leaf=8, min_samples_split=4, n_estimators=100) # Optimized using TPOT
cl8 = MLPClassifier(activation = "relu", alpha = 0.1, hidden_layer_sizes = (10,10,10),
                            learning_rate = "constant", max_iter = 2000, random_state = RANDOM_SEED)


# For this test I use Logistic Regression as a meta-classifier but you can ... take end experiment something else ...
mlr = LogisticRegression()
# Use classifiers from the list and build stacking cross validated classifier with meta-classifier on top (Logistic Regression, SVC ...)
# For this experiment I take only three of them.

# --- Tested during experiments --- (all classifiers)
# cl1 - [KNN] - Accuracy: 0.70  - AUC: 0.682
# cl2 - [RandomForest] - Accuracy: 0.88 - AUC: 0.931
# cl3 - [GaussianNB] - Accuracy: 0.79 - AUC: 0.871
# cl4 - [DecisionTree] - Accuracy: 0.88 - AUC: 0.920
# cl5 - [CatBoost] - Accuracy: 0.89 - AUC: 0.939
# cl6 - [LGBM] - Accuracy: 0.89 - AUC: 0.939
# cl7 - [ExtraTrees] - Accuracy: 0.89 - AUC: 0.939
# cl8 - [MLP] - Accuracy: 0.85 - AUC: 0.889

# "Ensemble learning works best when the base models are not correlated. 
# For instance, you can train different models such as linear models, decision trees, and neural nets on different datasets or features. 
# The less correlated the base models, the better." (https://neptune.ai/blog/ensemble-learning-guide)


scl = StackingCVClassifier(classifiers= [cl2, cl5, cl6, cl7], #[cl1, cl2, cl3, cl4, cl5, cl6, cl7, cl8]
                            meta_classifier = mlr, # use meta-classifier
                            use_probas = PROBAS,   # use_probas = True/False
                            random_state = RANDOM_SEED)
# All classifiers
#classifiers = {"KNN": cl1,
#               "RandomForest": cl2,
#               "GaussianNB": cl3,
#               "DecisionTree": cl4,
#               "CatBoost": cl5,
#               "LGBM": cl6,
#               "ExtraTrees": cl7,
#               "MLP": cl8}


# Number of classifiers used (define it according to classifiers used)
NUM_CLAS = 5 # classifiers (l1) + stacked (meta-classifier) 

# Classifiers for experiment + stacking (meta-classifier)
classifiers = {"RandomForest": cl2,
               "CatBoost": cl5,
               "LGBM": cl6,
               "ExtraTrees": cl7,
               "Stacked": scl}

X = all_df.drop([TARGET], axis = 1)
y = all_df[TARGET]

print (f'X:{X.shape} y: {y.shape} \n')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = RANDOM_SEED)
print (f'X_train:{X_train.shape} y_train: {y_train.shape}')
print (f'X_test:{X_test.shape} y_test: {y_test.shape}')

test = all_df[len(train_df):].drop([TARGET], axis = 1)
print (f'test:{test.shape}')

# This step could take some time .... it depends on classifiers you use .... So make a coffe or meditate ... 

print(">>>> Training started <<<<")
for key in classifiers:
    classifier = classifiers[key]
    scores = model_selection.cross_val_score(classifier, X_train, y_train, cv = FOLDS, scoring='accuracy')
    print("[%s] - accuracy: %0.2f " % (key, scores.mean()))
    classifier.fit(X_train, y_train)
    
    # Save classifier for prediction 
    classifiers[key] = classifier
    
# Tested during experiments --- (all classifiers)
# Accuracy: 0.70 [KNN]
# Accuracy: 0.88 [RandomForest]
# Accuracy: 0.79 [GaussianNB]
# Accuracy: 0.88 [DecisionTree]
# Accuracy: 0.89 [CatBoost]
# Accuracy: 0.89 [LGBM]
# Accuracy: 0.89 [ExtraTrees]
# Accuracy: 0.85 [MLP]

# Let's see how the models work ... We will operate on probas ...

preds = pd.DataFrame()

for key in classifiers:
    y_pred = classifiers[key].predict_proba(X_test)[:,1]
    preds[f"{key}"] = y_pred
    auc = metrics.roc_auc_score(y_test, y_pred)
    print(f"{key} -> AUC: {auc:.3f}")

preds[TARGET] = pd.DataFrame(y_test).reset_index(drop=True)

# Tested during experiments --- (all classifiers)
# KNN -> AUC: 0.682
# RandomForest -> AUC: 0.931
# GaussianNB -> AUC: 0.871
# DecisionTree -> AUC: 0.920
# CatBoost -> AUC: 0.939
# LGBM -> AUC: 0.939
# ExtraTrees -> AUC: 0.939
# MLP -> AUC: 0.889

# Seaborn will now show us how the models predict survival ...

sns.set(font_scale = 1)
sns.set_style({"axes.facecolor": "1.0", "axes.edgecolor": "0.85", "grid.color": "0.85",
               "grid.linestyle": "-", 'axes.labelcolor': '0.4', "xtick.color": "0.4",
               'ytick.color': '0.4'})

f, ax = plt.subplots(figsize=(13, 4), nrows=1, ncols = NUM_CLAS)

for key, counter in zip(classifiers, range(NUM_CLAS)):
    
    y_pred = preds[key]
   
    auc = metrics.roc_auc_score(y_test, y_pred)
    textstr = f"AUC: {auc:.3f}"


    false_pred = preds[preds[TARGET] == 0]
    sns.distplot(false_pred[key], hist=True, kde=True, 
                 bins=int(50), color = 'red', 
                 hist_kws={'edgecolor':'black'}, ax = ax[counter])
    

    true_pred = preds[preds[TARGET] == 1]
    sns.distplot(true_pred[key], hist=True, kde=True, 
                 bins=int(50), color = 'green', 
                 hist_kws={'edgecolor':'black'}, ax = ax[counter])
    
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    
    ax[counter].text(0.05, 0.95, textstr, transform=ax[counter].transAxes, fontsize=14,
                    verticalalignment = "top", bbox=props)
    
    ax[counter].set_title(f"{key}")
    ax[counter].set_xlim(0,1)
    ax[counter].set_xlabel("Probability")

plt.tight_layout()

shap.initjs()

# Lets explain how the models see the competition world ... What features are important .... What values drive the model for survival ...
# I made it only for one fast model (LGBM). Feel free to understand other models ...


explainer = shap.TreeExplainer(classifiers["LGBM"])
shap_values = explainer.shap_values(X_train)
shap.summary_plot(shap_values[1], X_train)

# And we approaching to the final stage ... prediction ... 

test_preds = classifiers['Stacked'].predict_proba(test)[:,1]

# Grandmaster tip -> Alexander Ryzhkov
# They way of finding "the best" from "the best" :) that is, secret codes for the game ...

threshold = pd.Series(test_preds).sort_values(ascending = False).head(34911).values[-1]
print(f"Current threshold is: {threshold}")

submission['submit_1'] = (test_preds > threshold).astype(int)
submission['submit_1'].mean()

# Next Grandmaster tip -> BIZEN
# Hacking the system :) How about mixing it with another submissions

submission['submit_2'] = pd.read_csv("./Kaggle/data/dae.csv")[TARGET]
submission['submit_3'] = pseudo_labels[TARGET]

submission[[col for col in submission.columns if col.startswith('submit_')]].sum(axis = 1).value_counts()

submission[TARGET] = (submission[[col for col in submission.columns if col.startswith('submit_')]].sum(axis=1) >= 2).astype(int)
submission[TARGET].mean()

# and now .... hold your breath and upload the results on the server and wait for the results ... TOP? How much?

submission[['PassengerId', TARGET]].to_csv("./Kaggle/data/other_submission.csv", index = False)