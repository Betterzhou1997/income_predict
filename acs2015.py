import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from preprocess import fill_nan, filter_by_f_regress, train

# 读取CSV文件

data = pd.read_csv('data/acs2015_census_tract_data.csv')

new_df = pd.DataFrame()
new_df['data'] = data['State'] + '-' + data['County']

# 查看缺失值
nan_counts = data.isnull().sum()
print(nan_counts)

nan_rows = data[data.isnull().any(axis=1)]

# 将包含 NaN 值的行保存到 CSV 文件
nan_rows.to_csv('nan_rows_acs2015.csv', index=False)
exit()

# 拼接State和County列

# 删除原始的State和County列
labels = data['Income'].values
# data.drop(['id', 'State', 'County', 'Income', 'Men', 'Women', 'Hispanic','White', 'Black'], axis=1, inplace=True)
data.drop(['id', 'State', 'County', 'Income'], axis=1, inplace=True)

feature = fill_nan(data.values)

print("初始特征维度为：", feature.shape)
# feature = filter_by_f_regress(feature, labels)
# print("f_regress后维度为：", feature.shape)


# 对State-County列进行one-hot编码
one_hot = 0
if one_hot:
    encoder = OneHotEncoder(sparse=False)
    encoded_features = encoder.fit_transform(new_df[['data']])
    print("One-hot编码维度为：", encoded_features.shape)
    feature = np.concatenate((feature, encoded_features), axis=1)
    print("最终的特征维度为：", feature.shape)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(feature, labels,
                                                    test_size=0.2,
                                                    random_state=666)

print("数据处理结束")

eval_set = [(X_test, y_test)]

best_model = train(X_train, y_train, eval_set, search=False)
# 在训练集上进行预测
train_predictions = best_model.predict(X_train)
#
# 在测试集上进行预测
test_predictions = best_model.predict(X_test)

train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))

print(train_rmse, test_rmse)
