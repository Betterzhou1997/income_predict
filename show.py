import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from preprocess import fill_nan, filter_by_f_regress, train
matplotlib.use('TkAgg')
# 读取CSV文件

data = pd.read_csv('data/train.csv')

new_df = pd.DataFrame()
new_df['data'] = data['State'] + '-' + data['County']

# 查看缺失值
nan_counts = data.isnull().sum()
print(nan_counts)
nan_rows = data[data.isnull().any(axis=1)]

# 将包含 NaN 值的行保存到 CSV 文件
nan_rows.to_csv('nan_rows_train.csv', index=False)

# 拼接State和County列

# 删除原始的State和County列
labels = data['Income'].values
# data.drop(['id', 'State', 'County', 'Income', 'Men', 'Women', 'Hispanic','White', 'Black'], axis=1, inplace=True)
data.drop(['id', 'State', 'County', 'Income'], axis=1, inplace=True)

feature = fill_nan(data.values)

print("初始特征维度为：", feature.shape)
print("标签维度为：", labels.shape)
X_train, X_test, y_train, y_test = train_test_split(feature, labels,
                                                    test_size=0.01,
                                                    random_state=666)
for i in range(30):
    plt.figure()
    plt.scatter(X_test[:, i], y_test, marker='.')
    plt.xlabel('Feature {}'.format(i+1))
    plt.ylabel('Label')
    plt.title('Scatter Plot of Feature {} and Label'.format(i+1))

plt.show()


