import pandas as pd

data = pd.read_csv('./project/mini/data/fma_genres_mfcc_000.csv', header=0)
print(data)

for i in range(1,156):
    data_ = pd.read_csv('./project/mini/data/fma_genres_mfcc_{}.csv'.format(str(i).zfill(3)), header=0)
    # 합치기
    data = pd.concat([data, data_], ignore_index=True)

print(data)
data.to_csv('./project/mini/data/fma_genres_mfcc.csv', index=False)

gtzan = pd.read_csv('./project/mini/data/new_genre_mfcc.csv', header=0)
fma = pd.read_csv('./project/mini/data/fma_genres_mfcc.csv', header=0)
print(gtzan.shape)      # (1200, 25)
print(fma.shape)        # (4994, 25)

df = pd.concat([gtzan, fma], ignore_index=True)
df.sort_values(by=['labels'], ignore_index=True, inplace=True)

df.to_csv('./project/mini/data/total_genres_mfcc.csv', index=False)

print(df.shape) # (6194, 25)

print(df['labels'].value_counts())

# rock          1198
# hiphop        1101
# pop           1096
# folk          1000
# electronic     999
# disco          100
# country        100
# dance          100
# classical      100
# jazz           100
# blues          100
# ballad         100
# reggae         100
