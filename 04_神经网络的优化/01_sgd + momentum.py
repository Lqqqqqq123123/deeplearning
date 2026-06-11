import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim

# 定义目标函数: 模拟的是损失函数, 参数是二维的向量
def f(w) -> torch.Tensor:
    return (w ** 2).dot(torch.tensor([0.05, 1]))

def grad_desc(w, optimizer, iters_num):
    w0_list = []  # 存储的值, 方便绘图
    w1_list = []

    for i in range(iters_num):
        w0_list.append(w[0].item())  # 保存参数值, 用于画图
        w1_list.append(w[1].item())

        # 1. 计算损失
        loss = f(w)
        # 2. 反向传播
        loss.backward()
        # 3. 更新参数
        optimizer.step()
        # 4. 梯度清零
        optimizer.zero_grad()

    return w0_list, w1_list

if __name__ == '__main__':
    # 1. 初始化参数
    w = torch.tensor([-7.0, 2])

    # 2. 定义超参数
    lr = 0.01
    epoch_num = 500

    # 3. 定义优化器

    # 3.1 普通的SGD优化器
    x_clone = w.clone().detach().requires_grad_(True)
    optimizer = optim.SGD([x_clone], lr=lr)
    x0_list, x1_list = grad_desc(x_clone, optimizer, epoch_num)

    plt.plot(x0_list, x1_list, color='red', label='SGD')

    # 3.2 动量优化器
    x_clone = w.clone().detach().requires_grad_(True)
    optimizer = optim.SGD([x_clone], lr=lr, momentum=0.9)
    x0_list, x1_list = grad_desc(x_clone, optimizer, epoch_num)

    plt.plot(x0_list, x1_list, color='blue', label='Momentum')

    # 绘制等高线
    ## 生成网格采样点
    w0_grid, w1_grid = np.meshgrid(np.linspace(-7, 7, 100), np.linspace(-2, 2, 100))
    y_grid = 0.05 * w0_grid ** 2 + w1_grid ** 2  # 计算y
    plt.contour(w0_grid, w1_grid, y_grid, levels=30, colors='gray')  # levels 等高线的数量

    plt.legend()
    plt.show()