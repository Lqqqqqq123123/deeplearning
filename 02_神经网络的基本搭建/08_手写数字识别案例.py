import warnings

import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

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
    dataset = pd.read_csv("data/train.csv")

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
    x_train = torch.tensor(x_train).float()       # 特征用 float32
    x_test = torch.tensor(x_test).float()
    y_train = torch.tensor(y_train.to_numpy()).long()  # 分类标签用 long (int64)
    y_test = torch.tensor(y_test.to_numpy()).long()

    return x_train, x_test, y_train, y_test

# 使用 Sequential 定义模型，与 nn_example.pt 的保存结构一致
model = nn.Sequential(
    nn.Linear(784, 50),   # 索引 0
    nn.ReLU(),            # 索引 1
    nn.Linear(50, 100),   # 索引 2
    nn.ReLU(),            # 索引 3
    nn.Linear(100, 10),   # 索引 4
)


if __name__ == '__main__':
    # 训练过程省略，直接用保存好的模型参数
    model.load_state_dict(torch.load('model/nn_example.pt', weights_only=True))
    model = model.to(device)  # 将模型移到 GPU

    # 测试
    x_train, x_test, y_train, y_test = get_digit_data()
    # 将数据和标签移到 GPU
    x_train = x_train.to(device)
    x_test = x_test.to(device)
    y_train = y_train.to(device)
    y_test = y_test.to(device)

    # 预测
    y_pred = model(x_test)
    # print(f'模型原始输出:{y_pred}')
    # print("输出形状：", y_pred.shape)  # 形状为(测试集样本数, 10)，对应每个样本的10个类别概率

    # 拿到最大概率
    y_pred_class = torch.argmax(y_pred, dim=-1)

    print("预测类别：\n", y_pred_class)

    # 评估
    correct = (y_pred_class == y_test).sum().item() / len(y_test)
    print(f'准确率为{correct:.2f}')





