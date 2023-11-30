import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from ml.preprocess import data_process_pipeline


# 创建自定义数据集类
class MyDataset(Dataset):
    def __init__(self, path, test=False, one_hot=False):
        columns, data, labels = data_process_pipeline(path='../data/train.csv', one_hot=True)
        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=123)
        if test:
            self.data = x_test
            self.label = y_test
        else:
            self.data = x_train
            self.label = y_train

    def __len__(self):
        # 返回数据集的大小
        return len(self.data)

    def __getitem__(self, idx):
        # 根据索引返回数据和标签
        x = self.data[idx]
        y = self.label[idx]
        x, y = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

        return x, y


if __name__ == '__main__':

    # 创建数据集实例
    dataset = MyDataset('../data/train.csv')

    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    for x, y in dataloader:
        print(x.shape, y.shape)
