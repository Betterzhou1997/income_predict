# 创建模型实例
import math

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from dl.load_data import MyDataset
from dl.model import MyModel, MyModelOnehot

if __name__ == '__main__':
    one_hot = True
    if not one_hot:
        model = MyModel()
    else:
        model = MyModelOnehot()

    # 定义MSE损失函数和Adam优化器
    criterion = nn.MSELoss()  # MSE损失函数即均方根误差的平方
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.5)

    # 创建数据集实例
    train_dataset = MyDataset('../data/train.csv', one_hot=one_hot)
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    # 创建数据集实例
    test_dataset = MyDataset('../data/train.csv', test=True, one_hot=one_hot)
    # 创建数据加载器
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    for epoch in range(100):

        train_rmse = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            output = model(x)
            y = y.unsqueeze(1)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            train_rmse += math.sqrt(criterion(output, y).item())
        train_rmse /= len(train_loader)

        test_rmse = 0.0
        for x_test, y_test in test_loader:
            output_test = model(x_test)
            test_rmse += math.sqrt(criterion(output_test, y_test.unsqueeze(1)).item())
        test_rmse /= len(test_loader)

        print(f'Epoch {epoch + 1}, Train RMSE: {train_rmse}, Test RMSE: {test_rmse}')
