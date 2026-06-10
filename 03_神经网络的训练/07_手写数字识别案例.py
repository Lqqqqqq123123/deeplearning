import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import pandas as pd

from torch.utils.data import DataLoader, TensorDataset, Dataset

# 设备检测：有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备: {device}')


def get_digit_data():
    """
    加载手写数字数据集，返回划分好、归一化后的 PyTorch 张量。

    Returns:
        x_train (tensor): 训练集特征，形状 (n_train, 784)，元素范围 [0, 1]
        x_test  (tensor): 测试集特征，形状 (n_test, 784)，元素范围 [0, 1]
        y_train (tensor): 训练集标签，形状 (n_train,)，取值 0~9
        y_test  (tensor): 测试集标签，形状 (n_test,)，取值 0~9
    """
    # 1. 从 CSV 加载原始数据集（每行一张 28×28=784 像素图片 + 1 列标签）
    dataset = pd.read_csv("../data/train.csv")

    # 2. 分离特征矩阵 x 和标签向量 y
    x = dataset.drop("label", axis='columns')  # (n_samples, 784)  所有像素灰度值 0~255
    y = dataset["label"]               # (n_samples,)      数字 0~9

    # 3. 按 8:2 随机划分为训练集和测试集（random_state 固定种子，结果可复现）
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    # 4. MinMax 归一化：将每列的像素值线性缩放到 [0, 1]
    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)  # 在训练集上学习 min/max → 再转换
    x_test = scaler.transform(x_test)        # 复用训练集的 min/max 转换测试集（防止数据泄露）

    # 5. 将 pandas 数据转为 PyTorch 张量
    x_train = torch.tensor(x_train, device=device).float()       # 特征用 float32
    x_test = torch.tensor(x_test, device=device).float()
    y_train = torch.tensor(y_train.to_numpy(), device=device).long()  # 分类标签用 long (int64)
    y_test = torch.tensor(y_test.to_numpy(), device=device).long()

    return x_train, x_test, y_train, y_test


# 超参数
lr = 0.01
batch_size = 64
epochs = 32

if __name__ == '__main__':
    x_train, x_test, y_train, y_test = get_digit_data()
    # 封装成 dataset
    train_dataset = TensorDataset(x_train, y_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(x_test, y_test)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f'训练集大小:{len(train_dataset)}, 测试集大小:{len(test_dataset)}')

    # 创建模型
    model = nn.Sequential(
        nn.Linear(784, 50),
        nn.ReLU(),
        nn.Linear(50, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )

    # 将模型和数据移动到device
    model = model.to(device)

    # 创建损失韩式和优化器
    ce_loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 开始训练
    for i in range(epochs):
        # 训练模式
        model.train()
        train_loss = 0
        train_acc_num = 0

        for input, target in train_dataloader:
            # 1. 前向传播
            y_pred = model(input)

            # 2. 计算损失
            loss = ce_loss(y_pred, target)

            # 3. 反向传播
            loss.backward()

            # 4. 更新参数
            optimizer.step()

            # 5. 清零梯度
            optimizer.zero_grad()

            # 6. 累计损失
            train_loss += loss.item() * (input.shape[0])

            # 累计预测准确数
            y_pred_class = torch.argmax(y_pred, dim=1)
            train_acc_num += torch.sum(y_pred_class == target).item()

            # 计算每轮的平均损失和准确率
        this_train_loss = train_loss / len(train_dataset)
        this_train_acc = train_acc_num / len(train_dataset)

        # 验证
        model.eval()  # 让模型处于验证模式
        val_loss = 0
        val_acc_num = 0
        for inputs, targets in test_dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            y_pred = model(inputs)
            loss = ce_loss(y_pred, targets)

            # 总损失
            val_loss += loss.item() * inputs.shape[0]
            # 累计预测准确数
            y_pred_class = torch.argmax(y_pred, dim=1)
            val_acc_num += torch.sum(y_pred_class == targets).item()

        this_val_loss = val_loss / len(test_dataset)
        this_val_acc = val_acc_num / len(test_dataset)

        print(
            f'第{i + 1}轮, 训练损失: {this_train_loss:.4f}, 训练准确率: {this_train_acc:.4f}, 验证损失: {this_val_loss:.4f}, 验证准确率: {this_val_acc:.4f}')






