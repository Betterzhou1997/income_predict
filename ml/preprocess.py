import numpy as np
from sklearn.feature_selection import f_regression
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import SVR
from ml.load_data import load_data


# https://www.heywhale.com/mw/dataset/60df1cf73aeb9c0017b925c1/file


def data_process_pipeline(path, one_hot=False):
    columns, X, country, y = load_data(path)

    feature = fill_nan(X)

    print("初始特征维度为：", feature.shape)
    feature = filter_by_f_regress(feature, y)
    print("f_regress后维度为：", feature.shape)

    scaler = StandardScaler()
    feature = scaler.fit_transform(feature)

    if one_hot:
        feature = one_hot_emb(feature, country)
    return columns, feature, y


def one_hot_emb(X, country):
    encoder = OneHotEncoder(sparse=False)
    encoded_features = encoder.fit_transform(country[['data']])
    print("One-hot编码维度为：", encoded_features.shape)
    feature = np.concatenate((X, encoded_features), axis=1)
    print("最终的特征维度为：", feature.shape)
    return feature


def filter_by_f_regress(x, y):
    f_values, p_values = f_regression(x, y)
    # print(f_values)
    # print(p_values)

    # 设置阈值
    f_threshold = 100
    p_threshold = 0.05

    # 选择同时满足条件的特征
    selected_features = x[:, (f_values > f_threshold) & (p_values < p_threshold)]

    return selected_features


def fill_nan(X, strategy='mean'):
    imputer = SimpleImputer(strategy=strategy)

    features_filled = imputer.fit_transform(X)

    return features_filled


def train(X, y, eval_set, search=False, choose='xgb'):
    if not search:

        if choose == 'xgb':
            model = xgb.XGBRegressor(n_jobs=-1,
                                     # subsample=0.5
                                     )
            model.fit(X, y,
                      # eval_set=eval_set, verbose=True,
                      # eval_metric="rmse",
                      )
            return model


    else:
        # 定义XGB回归模型

        cv_params = {'n_estimators': [200, 300, 400, 500, 600, 700, 800]}
        other_params = {'learning_rate': 0.1, 'n_estimators': 500,
                        'max_depth': 5,
                        'min_child_weight': 1, 'seed': 0, 'subsample': 0.8,
                        'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0,
                        'reg_lambda': 1, 'n_jobs': -1}

        model = xgb.XGBRegressor(**other_params)
        optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params,
                                     scoring='neg_root_mean_squared_error',
                                     cv=5, verbose=5, n_jobs=-1)
        optimized_GBM.fit(X, y)

        print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
        print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
        best_model = optimized_GBM.best_estimator_
        return best_model
