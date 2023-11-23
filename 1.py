import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from preprocess import fill_nan, filter_by_f_regress
from sklearn.model_selection import GridSearchCV

# 读取CSV文件

data = pd.read_csv('data/train.csv')

new_df = pd.DataFrame()
new_df['data'] = data['State'] + '-' + data['County']

# 查看缺失值
nan_counts = data.isnull().sum()
print(nan_counts)
nan_rows = data[data.isnull().any(axis=1)]

# 将包含 NaN 值的行保存到 CSV 文件
# nan_rows.to_csv('nan_rows.csv', index=False)


# 拼接State和County列

# 删除原始的State和County列
labels = data['Income'].values
data.drop(['id', 'State', 'County', 'Income'], axis=1, inplace=True)

feature = fill_nan(data.values)

print("初始特征维度为：", feature.shape)
feature = filter_by_f_regress(feature, labels)
print("f_regress后维度为：", feature.shape)

# 对State-County列进行one-hot编码

encoder = OneHotEncoder(sparse=False)
encoded_features = encoder.fit_transform(new_df[['data']])
print("One-hot编码维度为：", encoded_features.shape)
final_x = np.concatenate((feature, encoded_features), axis=1)
print("最终的特征维度为：", final_x.shape)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(final_x, labels,
                                                    test_size=0.2,
                                                    random_state=666)

print("数据处理结束")
# 定义XGB回归模型
model = xgb.XGBRegressor(
    n_jobs=-1,
    #                      learning_rate=0.01,
    #                      n_estimators=200,
    #                      verbose=10,
    #                      objective='reg:linear',
    #                      reg_alpha=0,
    #                      reg_lambda=1
                         )
print(model)

# cv_params = {'n_estimators': [400, 500, 600, 700, 800]}
# other_params = {'learning_rate': 0.1, 'n_estimators': 500, 'max_depth': 5,
#                 'min_child_weight': 1, 'seed': 0, 'subsample': 0.8,
#                 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0,
#                 'reg_lambda': 1}
#
# model = xgb.XGBRegressor(**other_params)
# optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params,
#                              scoring='rmse', cv=5, verbose=1, n_jobs=4)
# optimized_GBM.fit(X_train, y_train)

# 训练模型
model.fit(X_train, y_train)

# optimized_GBM.fit(X_train, y_train)
#
# print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
# print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))

# 在训练集上进行预测
train_predictions = model.predict(X_train)

# 在测试集上进行预测
test_predictions = model.predict(X_test)

train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))

print(train_rmse, test_rmse)
