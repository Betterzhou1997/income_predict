from sklearn.model_selection import train_test_split
from preprocess import data_process_pipeline
from train_utils import get_result

columns, x, y = data_process_pipeline(path='data/train.csv', one_hot=False)
print(columns)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123)
train_rmse, test_rmse = get_result(x_train, y_train, x_test, y_test)
print('train_rmse: {:.1f}, test_rmse: {:.1f}'.format(train_rmse, test_rmse))