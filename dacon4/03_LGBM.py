import numpy as np 
import pandas as pd 
# import math
import os  # 디렉토리 변경
from sklearn.model_selection import GridSearchCV 
from sklearn.metrics import make_scorer  # loss function 커스터마이징

from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split 

os.chdir('./dacon4/data') 

train=pd.read_csv('train.csv', encoding='euc-kr')
test=pd.read_csv('test.csv', encoding='euc-kr')
submission=pd.read_csv('sample_submission.csv', encoding='euc-kr')

#건물별로 '비전기냉방설비운영'과 '태양광보유'를 판단해 test set의 결측치를 보간해줍니다
train[['num', '비전기냉방설비운영','태양광보유']]
ice={}
hot={}
count=0
for i in range(0, len(train), len(train)//60):
    count +=1
    ice[count]=train.loc[i,'비전기냉방설비운영']
    hot[count]=train.loc[i,'태양광보유']

for i in range(len(test)):
    test.loc[i, '비전기냉방설비운영']=ice[test['num'][i]]
    test.loc[i, '태양광보유']=hot[test['num'][i]]


# 시간, 요일, 주말여부(new!) 추가
def time(x):
    return int(x[-2:])
train['time']=train['date_time'].apply(lambda x: time(x))
test['time']=test['date_time'].apply(lambda x: time(x))

# 평일=0~4, 주말=5~6
def weekday(x):
    return pd.to_datetime(x[:10]).weekday()
train['weekday']=train['date_time'].apply(lambda x :weekday(x))
test['weekday']=test['date_time'].apply(lambda x :weekday(x))

# 평일=0, 주말=1
train['weekend']=train['weekday'].apply(lambda x: 0 if x < 4 else 1)
test['weekend']=test['weekday'].apply(lambda x: 0 if x < 4 else 1)


# 기온, 풍속, 습도 등, 기타 결측치를 적당히 1/3, 2/3 수치로 보간해줍니다.
test = test.interpolate(method='values')  


# 모델링
# 학습용set 생성
train.drop('date_time', axis=1, inplace=True)  # 학습에 불필요한 날짜 제거
train_x=train.drop('전력사용량(kWh)', axis=1)  # 문제
train_y=train[['전력사용량(kWh)']]  # 정답

X_train, X_valid, y_train, y_valid = train_test_split(train_x, train_y, test_size=0.2, random_state=156)


# loss function : SMAPE 정의
# from sklearn.metrics import mean_absolute_error
def smape(true, pred):
    true = np.array(true)  # np.array로 바꿔야 에러 없음
    pred = np.array(pred)
    return np.mean((np.abs(true-pred))/(np.abs(true) + np.abs(pred)))  # *2 , *100은 상수이므로 생략
SMAPE = make_scorer(smape, greater_is_better=False)  # smape 값이 작아져야하므로 False


# 파라미터 설정, 모델생성 함수
def get_best_params(model, params):
    grid_model = GridSearchCV(
        model,
        param_grid = params,  # 파라미터
        cv=5,  # Kfold : 5
        scoring= SMAPE)  #loss function

    grid_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)], verbose=100)
    scr = grid_model.best_score_
    print(f'{model.__class__.__name__} 최적 score 값 {scr}')
    return grid_model.best_estimator_


# 파라미터 후보군 설정
# 어떤 파라미터로 하는게 좋을지 고민된다면 고민하는 것들을 리스트 안에 다 넣어보세요 알아서 골라줄겁니다.
# 저는 예시로 learning_rate만 0.1 or 0.01 중 더 좋은걸 골라달라고 했습니다.
params = {}
params['boosting_type'] = ['gbdt']
params['objective'] = ['regression']
params['n_estimators'] = [100]
params['learning_rate'] = [0.1, 0.01]  
params['subsample'] = [1]


# 모델정의
model=LGBMRegressor(params)


# 학습진행
best_lgbm = get_best_params(model, params)
best_lgbm  # learning_rate 0.1 or 0.01 중, 0.01이 더 좋았다고 하는군요. 


# 기타 지표로 에러 측정
from sklearn.metrics import mean_squared_error, r2_score
y_pred = best_lgbm.predict(X_train)

mse_score = mean_squared_error(y_train, y_pred)
r2_score = r2_score(y_train, y_pred)
print('MSE:', mse_score)
print('R2 :', r2_score)


# 모델저장, 로드
from sklearn.externals import joblib
joblib.dump(best_lgbm, 'best_lgbm.pkl')
load_lgbm = joblib.load('best_lgbm.pkl')


# 모델에 넣기 위해 날짜칼럼 제거
test_x = test.drop('date_time', axis=1)


# 모델 예측
submission_y = best_lgbm.predict(test_x)


# submission.csv 생성
test['answer'] = submission_y
test['num_date_time'] = test.apply(lambda x: str(x['num']) +' '+ x['date_time'], axis=1)
submission = test[['num_date_time', 'answer']]
submission.to_csv('submission.csv', index=False)
print(submission)