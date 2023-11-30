import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# 定义1维卷积神经网络模型


class Residual(torch.nn.Module):
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = torch.nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.b1 = torch.nn.BatchNorm1d(out_channels)
        self.b2 = torch.nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = F.leaky_relu(self.b1(self.conv1(x)))
        x = self.b2(self.conv2(x))
        if self.conv3 is not None:
            x = self.conv3(x)
        return F.leaky_relu(x + x)


class MyModel(nn.Module):

    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3),
            nn.BatchNorm1d(16),
            nn.Conv1d(in_channels=16, out_channels=64, kernel_size=3),
        )
        self.res1 = Residual(in_channels=64, out_channels=64, use_1x1conv=True)
        self.res2 = Residual(in_channels=64, out_channels=64, use_1x1conv=True)

        self.fc = nn.Sequential(nn.Linear(64 * 26, 128),
                                nn.LeakyReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(128, 32),
                                nn.LeakyReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(32, 1)
                                )

    def forward(self, x):
        x = x.unsqueeze(1)
        # 增加通道数
        x = self.conv1(x)
        x = F.leaky_relu(x)
        # 残差块
        x = self.res1(x)
        x = self.res2(x)
        x = x.view(x.size(0), -1)  # 展开为一维向量
        x = self.fc(x)
        return x


class MyModelOnehot(nn.Module):

    def __init__(self):
        super(MyModelOnehot, self).__init__()
        self.one_hot = nn.Sequential(
            nn.Linear(3115 - 30, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, 64),

        )
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3),
            nn.BatchNorm1d(16),
            nn.Conv1d(in_channels=16, out_channels=64, kernel_size=3),
        )
        self.res1 = Residual(in_channels=64, out_channels=64, use_1x1conv=True)
        self.res2 = Residual(in_channels=64, out_channels=64, use_1x1conv=True)

        self.fc = nn.Sequential(nn.Linear(64 * 26, 128),
                                nn.LeakyReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(128, 32),
                                nn.LeakyReLU(),
                                nn.Dropout(0.5),
                                nn.Linear(32, 1)
                                )

    def forward(self, x):
        x1 = x[:, 30]
        x2 = x[:, 30:]
        print(x1.shape)
        print(x2.shape)
        x1 = x1.unsqueeze(1)
        # 增加通道数
        x1 = self.conv1(x1)
        x1 = F.leaky_relu(x1)
        # 残差块
        x1 = self.res1(x1)
        x1 = self.res2(x1)
        x1 = x1.view(x1.size(0), -1)  # 展开为一维向量
        x1 = self.fc(x1)
        return x1


if __name__ == '__main__':
    # 创建模型实例
    model = MyModelOnehot()
    # 定义RMSE损失函数和Adam优化器
    criterion = nn.MSELoss()  # RMSE损失函数即均方根误差的平方
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    input_data = torch.randn(10, 3115)  # 生成一个大小为(10, 30)的随机输入数据
    target = torch.randn(10, 1)  # 生成一个大小为(10, 1)的随机目标值
    for epoch in range(20):
        optimizer.zero_grad()
        output = model(input_data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        print(loss.item())
