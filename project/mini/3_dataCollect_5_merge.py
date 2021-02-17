# GTZAN 과 FMA 에서 추출한 데이터 합치기!!

import numpy as np
import pandas as pd

gtzan = pd.read_csv('./project/mini/data/gtzan_genres_mfcc.csv', header=0)
fma = pd.read_csv('./project/mini/data/fma_genres_mfcc.csv', header=0)
print(gtzan.shape)
print(fma.shape)

df = pd.concat([gtzan, fma], ignore_index=True)
df.sort_values(by=['labels'], ignore_index=True, inplace=True)

df.to_csv('./project/mini/data/total_genres_mfcc.csv', index=False)

print(df)

print(df['labels'].value_counts())