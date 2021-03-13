import pandas as pd

data = pd.read_csv('./project/mini/data/fma_genres_mfcc_000.csv', header=0)
print(data)

for i in range(1,156):
    data_ = pd.read_csv('./project/mini/data/fma_genres_mfcc_{}.csv'.format(str(i).zfill(3)), header=0)
    # 합치기
    data = pd.concat([data, data_], ignore_index=True)

print(data)
data.to_csv('./project/mini/data/fma_genres_mfcc.csv', index=False)
