import xgboost as xgb
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error


def train(x, y, eval_set, search=False, choose='xgb'):
    if not search:

        if choose == 'xgb':
            model = xgb.XGBRegressor(n_jobs=-1,
                                     # subsample=0.5
                                     )
            model.fit(x, y,
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
        optimized_GBM.fit(x, y)

        print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
        print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
        best_model = optimized_GBM.best_estimator_
        return best_model


def get_result(x_train, y_train, x_test, y_test):
    eval_set = [(x_test, y_test)]
    best_model = train(x_train, y_train, eval_set, search=False)
    # 在训练集上进行预测
    train_predictions = best_model.predict(x_train)
    #
    # 在测试集上进行预测
    test_predictions = best_model.predict(x_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))


    return train_rmse, test_rmse


def evaluate_feature_combination(index, columns, x, y):
    index_columns = [columns[i] for i in index]
    x = x[:, index]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)
    train_rmse, test_rmse = get_result(x_train, y_train, x_test, y_test)
    return index_columns, train_rmse, test_rmse

