from sklearn.feature_selection import f_regression

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error


def filter_by_f_regress(X, y):
    f_values, p_values = f_regression(X, y)
    print(f_values)
    print(p_values)

    # 设置阈值
    f_threshold = 10
    p_threshold = 0.05

    # 选择同时满足条件的特征
    selected_features = X[:,
                        (f_values > f_threshold) & (p_values < p_threshold)]

    return selected_features


def fill_nan(X, strategy='mean'):
    imputer = SimpleImputer(strategy=strategy)

    features_filled = imputer.fit_transform(X)

    return features_filled


if __name__ == '__main__':
    data = pd.read_csv('data/train.csv')

    data.drop(['id', 'State', 'County'], axis=1, inplace=True)

    features = data.drop(['Income'], axis=1).values
