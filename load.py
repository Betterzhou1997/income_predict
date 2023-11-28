import pandas as pd


def load_data(path: str, print_Nan=False):
    data = pd.read_csv(path)

    if print_Nan:
        nan_counts = data.isnull().sum()
        print(nan_counts)
        # nan_rows = data[data.isnull().any(axis=1)]
        # 将包含 NaN 值的行保存到 CSV 文件
        # nan_rows.to_csv('nan_rows_train.csv', index=False)

    country = pd.DataFrame()
    country['data'] = data['State'] + '-' + data['County']

    labels = data['Income'].values

    data.drop(['id', 'State', 'County', 'Income'], axis=1, inplace=True)
    columns = data.columns
    return columns, data, country, labels


