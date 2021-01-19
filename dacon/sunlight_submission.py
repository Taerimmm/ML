# import numpy as np
# import pandas as pd

# from tensorflow.keras.models import load_model
# model = load_model('./dacon/data/sunlight_model.hdf5')

# submission = pd.DataFrame(np.zeros((7776,9)))
# submission.columns = pd.read_csv('./dacon/data/sample_submission.csv', index_col=0, header=0).columns
# submission.index = pd.read_csv('./dacon/data/sample_submission.csv', index_col=0, header=0).index
# print(submission)
# '''
# import re
# # print(list(submission.index))
# # print(list(re.findall('[0-9]+',i[2:]) for i in list(submission.index)))
# submission.iloc[:,0:3] = np.array(pd.DataFrame(re.findall('[0-9]+',i[2:]) for i in list(submission.index)))
# '''

# quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# for i in range(81):
#     # a = []
#     data = pd.read_csv("./dacon/data/test/{}.csv".format(i), header=0).set_index(['Day','Hour','Minute'])
#     for j, k in enumerate(quantiles):
#         a = []
#         a.append(np.array(data))
#         x_test = np.array(a)
#         y_pred = model.predict(x_test)
#         submission.iloc[96*i:96*(i+1), j] = np.array(y_pred)

# print(submission)
# print(submission.shape)

# submission.to_csv('./dacon/data/sample_submission_new.csv')


# # 분포 확인
# import matplotlib.pyplot as plt
# plt.plot(submission.iloc[:,-1][:192])
# plt.grid()
# plt.show()



# Quantile 해결
import numpy as np
import pandas as pd

quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

df = pd.DataFrame(np.array([[1, 1], [2, 10], [3, 100], [4, 100], [5, 6], [6, 8], [7,10], [8, 11], [9,12], [10 ,11]]),
                  columns=['a', 'b'])
df.quantile(.1)
print(df)
print()
print(df.quantile(0.1, interpolation='linear'))