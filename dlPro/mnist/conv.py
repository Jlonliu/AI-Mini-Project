import sys, os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms, datasets
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
from mnist.get_mnist import load_mnist

# 超参数定义
input_size = 28
num_classes = 10
num_epochs = 3
batch_size = 64
learning_rate = 0.01

# 数据集加载
(x_train, t_train), (x_valid, t_valid) = load_mnist()
# 将numpy数组转换成tensor
x_train = torch.from_numpy(x_train)
t_train = torch.from_numpy(t_train)
x_valid = torch.from_numpy(x_valid)
t_valid = torch.from_numpy(t_valid)

# 判断是否有GPU加速
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x_train, x_valid = x_train.to(device), x_valid.to(device)
t_train, t_valid = t_train.to(device), t_valid.to(device)

# 打包数据
train_ds = TensorDataset(x_train, t_train)
valid_ds = TensorDataset(x_valid, t_valid)
train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=batch_size)


# 定义模型
class TestCNN(nn.Module):

    def __init__(self):
        super(TestCNN, self).__init__()
        # nn.Sequential : 实现顺序连接模型
        self.layer1 = nn.Sequential(  # 输入(1,28,28)
            # nn.Conv2d : 2维卷积
            nn.Conv2d(
                in_channels=1,  # 通道数：1-灰度图
                out_channels=16,  # 要得到多少个特征图
                kernel_size=5,  # 卷积核大小
                stride=1,  # 步长
                padding=2  # 原图外围填充大小
            ),  # 输出16个28*28的特征图，记作(16,28,28)
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))  # 2*2区域的池化操作,输入(16,14,14)
        self.layer2 = nn.Sequential(  #输入(16,14,14)
            nn.Conv2d(16, 32, 5, 1, 2),  #输出(32,14,14)
            # nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))  # 输出(32,7,7)
        self.out = nn.Linear(32 * 7 * 7, 10)  # 输出32*7*7输出为10分类的全连接层

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)  # 展平: 重构张量的维度
        output = self.out(x)
        return output


model = TestCNN(input_size, num_classes)
model.to(device)

def accuarcy(pred, real):
    # 计算准确率
    # 1. 把预测值转换成one-hot形式
    pred = torch.argmax(pred, dim=1)
    # 2. 计算预测正确的样本个数
    # 3. 返回准确率
    return (pred == real).sum().item() / real.size(0)