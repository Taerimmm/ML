# 실습
# outlier1 을 행렬형태로 적용할 수 있도록 수정

import numpy as np

aaa = np.array([[1,2,3,4,10000,6,7,5000,90,100],
              [10,20,3,40,50,60,70,8,90,100]])

aaa = aaa.transpose()
print(aaa.shape)    # (10, 2)


def outliers(data_out):
    data = data_out.transpose()
    outlier = []
    for i in range(data.shape[0]):
        print(str(i) + '열')
        quartile_1, q2, quartile_3 = np.percentile(data[i], [25, 50, 75])
        print('1사분위 :', quartile_1)
        print('q2 :', q2)
        print('3사분위 :', quartile_3)
        iqr = quartile_3 - quartile_1
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)
        print('Boxplot 범위 :', lower_bound ,' ~ ', upper_bound)

        outlier.append(np.where((data[i] > upper_bound) | (data[i] < lower_bound)))
        print()
    return np.array(outlier)

outlier_loc = outliers(aaa)
print('이상치의 위치 :', outlier_loc)

import matplotlib.pyplot as plt

plt.boxplot(aaa)
plt.show()