import torch
import torch.nn as nn


# 自定义模型，必须继承nn.Module
class MyModule(nn.Module):
    def __init__(self):
        # 必须调用父类的初始化方法，完成Module的基础初始化
        super().__init__()

        # 定义网络层结构：对应图中的3个全连接层
        # 输入层→第1隐藏层：输入特征3，输出特征4
        self.linear1 = nn.Linear(in_features=3, out_features=4)
        # 第1隐藏层→第2隐藏层：输入特征4，输出特征4
        self.linear2 = nn.Linear(in_features=4, out_features=4)
        # 第2隐藏层→输出层：输入特征4，输出特征2（对应2分类任务）
        self.out = nn.Linear(in_features=4, out_features=2)

    # 必须实现forward方法，定义前向传播逻辑
    def forward(self, x):
        # 第1层：线性变换 + Tanh激活（对应要求）
        x = self.linear1(x)
        x = torch.tanh(x)

        # 第2层：线性变换 + ReLU激活（对应要求）
        x = self.linear2(x)
        x = torch.relu(x)

        # 输出层：线性变换 + Softmax激活，dim=-1表示对最后一维做概率归一化
        x = self.out(x)
        y = torch.softmax(x, dim=-1)

        # 返回最终输出
        return y


# 测试代码
if __name__ == '__main__':
    # 1. 构造输入数据：10个样本，每个样本3个特征（对应输入维度3）
    x = torch.randn(10, 3)

    # 2. 实例化自定义模型
    model = MyModule()

    # 3. 前向传播：自动调用forward方法，得到输出
    y = model(x)
    print("模型输出：\n", y)
    print("输出形状：", y.shape)  # 输出形状为(10,2)，对应10个样本、2个类别概率