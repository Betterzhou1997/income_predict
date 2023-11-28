import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from feature_select import get_all_index
from preprocess import train, data_process_pipeline

# 读取CSV文件


columns, X, y = data_process_pipeline(path='data/train.csv', one_hot=False)

best_rmse = 1000000000
best_features = []
for index in get_all_index(30):
    print('=======================================')
    print([columns[i] for i in index])
    X_train = X[:, index]
    # X_test, y_test = data_process_pipeline(path='data/merged_file.csv', one_hot=False)

    # print(X_test.shape, y_test.shape)
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_train, y,
                                                        test_size=0.2,
                                                        random_state=123)

    eval_set = [(X_test, y_test)]

    best_model = train(X_train, y_train, eval_set, search=False)
    # 在训练集上进行预测
    train_predictions = best_model.predict(X_train)
    #
    # 在测试集上进行预测
    test_predictions = best_model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, train_predictions))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))

    print('train_rmse: {:.1f}, test_rmse: {:.1f}'.format(train_rmse, test_rmse))
    if test_rmse < best_rmse:
        best_rmse = test_rmse
        best_features = [columns[i] for i in index]
        with open('output.txt', 'w') as file:
            file.write(' '.join(best_features))
            file.write('\n')
            file.write(str(best_rmse))