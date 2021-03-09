import pandas as pd
import matplotlib.pyplot as plt

wine = pd.read_csv('../data/csv/winequality-white.csv', header=0, sep=';')

count_data = wine.groupby('quality')
print(count_data)

count_data.plot()
plt.show()