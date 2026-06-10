import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset

batch_size = 6
epochs = 100

# 1. 准备数据
x = torch.randn(100, 1)
y = x * 2.5 + 5.0 + torch.randn(100, 1)

# 2. 构造数据加载器
dataset = TensorDataset(x, y)

dataloader = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=True
)

# 3. 搭建模型
model = nn.Linear(in_features=1, out_features=1)

# 4. 定义损失函数
loss_mse = nn.MSELoss()

# 5. 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 6. 训练
loss_list = []  # 存储每轮的平均损失

for i in range(epochs):
    # 遍历数据集
    total_loss = 0
    iter_num = 0
    for input, target in dataloader:
        # 每一个 batch 的数据，要做的就是 前向传播 + 反向传播 + 优化
        # 6.1 前向传播
        y_pred = model(input)
        # 6.2 计算损失
        loss = loss_mse(y_pred, target)
        # 6.3 反向传播
        loss.backward()
        optimizer.step()

        optimizer.zero_grad()
        total_loss += loss.item()

        iter_num += 1

    loss_list.append(total_loss / iter_num)

print('斜率：',model.weight)
print('偏置：',model.bias)

# 画图
import matplotlib.pyplot as plt

# 损失下降曲线
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].plot(loss_list)

# 散点图
axs[1].scatter(x, y)
# 拟合的直线
axs[1].plot(x, model(x).detach().numpy(), c='r')
plt.show()





