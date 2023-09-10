import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch import optim
# import pickle
import numpy as np
import cv2
import get_mnist as gm  # 获取mnist数据专用

# loss_func = F.cross_entropy  # 交叉熵函数


class MyTest(nn.Module):

    def __init__(self, xlen):
        """
        xlen是输入数据的长度，即输入数据的数值数量"""
        super().__init__()
        """
        Linear函数对输入进行线性变换后输出
        参数：
            in_features: 输入样本大小
            out_features: 输出样本大小
            bias: 是否附加偏置
        """
        self.hidden1 = nn.Linear(xlen, 128)  # 隐藏层1
        self.hidden2 = nn.Linear(128, 256)  # 隐藏层2
        self.out = nn.Linear(256, 10)  # 输出层
        """
        self.dropput是一个函数，这个函数将按0.5的概率随机丢弃输入的数据
        """
        self.dropput = nn.Dropout(0.5)  # 随机按概率丢弃输入数据的函数

    def forward(self, x):
        # x进行线性变换xw+b后通过激活函数获得输出
        h1 = F.relu(self.hidden1(x))
        # 对输入的数据进行随机丢弃数值（将数值变为0）
        h1_drop = self.dropput(h1)
        # 进行第二层的计算
        h2 = F.relu(self.hidden2(h1_drop))
        # 再次随机丢弃数值
        h2_drop = self.dropput(h2)
        # 进行最后的计算并输出结果
        y = self.out(h2_drop)
        return y


def get_data(train_ds, valid_ds, bs):
    """
    Dataloader函数，从给定数据集train_ds中，
    一次抽出给定bs个数的数据并返回，shuffle代表是否打乱数据集的数据顺序
    返回值是原始数据集中的一小部分（bs个）
    为什么要一次取出几个进行计算而不是直接计算整个数据集？
    因为如果数据集过于庞大，需要的显存就很大，分批次计算可以减轻显存压力
    也就是说如果你有超级无敌牛逼电脑，那你完全可以不使用DataLoader函数，
    直接使用TensorDataset函数的返回值
    """
    return (DataLoader(train_ds, batch_size=bs, shuffle=True), DataLoader(valid_ds, batch_size=bs))


def get_model(xlen=784):
    model = MyTest(xlen)
    """
    optim是一个实现了多种优化算法的包
    Adam利用梯度的一阶矩估计和二阶矩估计动态调整每个参数的学习率
    参数：
        params: 模型每一层的权重参数w和偏置参数b
        lr: 学习率
    """
    return model, optim.Adam(model.parameters(), lr=0.001)


def loss_batch(model, loss_func, x, t, opt=None):
    loss = loss_func(model(x), t)
    if opt is not None:
        # 反向计算损失
        loss.backward()
        # 沿梯度方向更新权重参数和偏置参数
        opt.step()
        # 将梯度值清零，以便不影响第二轮的梯度计算
        opt.zero_grad()
        # 返回值是 本次计算的损失的数值和本次计算的样本数量
    return loss.item(), len(x)


def fit(steps, model, loss_func, opt, train_dl, valid_dl):
    for step in range(steps):  # 训练次数
        model.train()  # 训练
        for x, t in train_dl:
            loss_batch(model, loss_func, x, t, opt)  # 更新权重参数

        model.eval()  # 验证训练结果
        with torch.no_grad():  # 不进行梯度计算
            # 返回值：一批次的损失，一批次的样本数量
            losses, nums = zip(*[loss_batch(model, loss_func, x, t) for x, t in valid_dl])
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        # 存储模型参数
        if val_loss < 0.07:
            torch.save(model.state_dict(), "./state_dict_step" + str(step) + "_loss" + str(val_loss) + ".pt")

        print("当前step: " + str(step), "验证集损失: " + str(val_loss))


# 接受一个输入数据，返回这个数据对应的标签
def Recognition_One_By_One(model, datum, label):  # 测试参数
    """
    参数：
        model: 训练后的模型
        datum: 一个输入数据（注意：只接受一个）
        label: 与输出数据对应的标签
        输出数据为[a,b,c]三个数据的话
        标签应该为[x,y,z]三个数据的、
        返回a、b、c中最大值对应的标签的值
        """
    model.load_state_dict(torch.load("./state_dict_step1182_loss0.004810350753564308.pt"))
    model.eval()
    with torch.no_grad():  # 不进行梯度计算
        y = model.forward(datum)
        # cv2.imshow("img", np.array(datum * 255, np.uint8).reshape(28, 28))
        # cv2.waitKey(0)
        return label[np.argmax(y).item()]

    #     # 返回值：一批次的损失，一批次的样本数量
    #     losses, nums = zip(*[loss_batch(model, loss_func, x, t) for x, t in valid_dl])
    # val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
    # print("验证集损失: " + str(val_loss))


if __name__ == "__main__":

    (x_train, t_train), (x_valid, t_valid) = gm.small_minist(1000)
    x_valid, t_valid = x_train, t_train
    # 将numpy数组转换成tensor
    # x_train, t_train, x_valid, t_valid = map(torch.tensor, (x_train, t_train, x_valid, t_valid))
    x_train = torch.from_numpy(x_train)
    t_train = torch.from_numpy(t_train)
    x_valid = torch.from_numpy(x_valid)
    t_valid = torch.from_numpy(t_valid)
    bs = 100
    """
    TensorDataset函数对输入的两个tensor数据进行打包：
    使两个数据的一维元素彼此对应
    在这里是让x_train中每一张图片（784个数值的tensor）与t_train中的每一个标签对应（0～9的int数值）
    """
    train_ds = TensorDataset(x_train, t_train)
    valid_ds = TensorDataset(x_valid, t_valid)

    train_dl, valid_dl = get_data(train_ds, valid_ds, bs)  # 获取数据
    model, opt = get_model(xlen=784)  # 获取模型
    fit(100, model, F.cross_entropy, opt, train_dl, valid_dl)  # 训练并验证
