import pandas as pd

data = pd.read_csv('./project/mini/data/total_genres_mfcc.csv')
print(data['labels'].value_counts())


'''
# 변경 전
hiphop           1101
rock             1098
pop              1096
instrumental     1000
folk             1000
electronic        999
international     996
experimental      983
metal             100
classical         100
blues             100
disco             100
country           100
jazz              100
reggae            100

- 15개genre -> 몇개 교체 dance랑 ballad 

experimental  제거
instrumental  제거
international 제거

rock - metal 합치기 (비슷한 부류)

# 변경 후

rock          1198
hiphop        1101
pop           1096
folk          1000
electronic     999
reggae         100
ballad         100
jazz           100
blues          100
country        100
dance          100
disco          100
classical      100
'''