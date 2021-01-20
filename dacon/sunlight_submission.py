import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model
model_2 = load_model('./dacon/data/sunlight_model_binary.hdf5')

submission = pd.DataFrame(np.zeros((7776,9)))
submission.columns = pd.read_csv('./dacon/data/sample_submission.csv', index_col=0, header=0).columns
submission.index = pd.read_csv('./dacon/data/sample_submission.csv', index_col=0, header=0).index
print(submission)

quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for i in range(81):
    # a = []
    data = pd.read_csv("./dacon/data/test/{}.csv".format(i), header=0).set_index(['Day','Hour','Minute'])
    for j, k in enumerate(quantiles):
        model_1 = load_model('./dacon/data/sunlight_model_{}_qauntile{}.hdf5'.format(j+1,k), compile=False)
        a = []
        a.append(np.array(data))
        x_test = np.array(a)
        y_pred_1 = model_1.predict(x_test)
        a = []
        a.append(np.array(data.iloc[:,-1:]))
        x_test = np.array(a)
        y_pred_2 = model_2.predict(x_test)
        y_pred_2 = np.array([1 if x > 0.5 else 0 for x in (y_pred_2).reshape(96)])
        submission.iloc[96*i:96*(i+1), j] = np.array([x for x in (y_pred_1*y_pred_2).reshape(96)])
        # submission.iloc[96*i:96*(i+1), j] = np.array([x if x > 1 else 0 for x in (y_pred_1*y_pred_2).reshape(96)])

print(submission)
print(submission.shape)

submission.to_csv('./dacon/data/sample_submission_new.csv')

'''
# 분포 확인
submission = pd.read_csv('./dacon/data/sample_submission_new.csv', index_col=0, header=0)
import matplotlib.pyplot as plt
plt.plot(submission.iloc[:,-1][:192])
plt.grid()
plt.show()
'''