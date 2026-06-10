import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib

# 解决中文乱码问题
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置中文字体（黑体）
matplotlib.rcParams['axes.unicode_minus'] = False     # 解决负号显示为方块的问题

# ============== 1. 准备数据 ==============
# 真实的函数关系: y = x^2
x_train = torch.linspace(-1, 1, 200).reshape(-1, 1)  # shape: (200, 1)
y_train = x_train ** 2                              # shape: (200, 1)

# ============== 2. 定义模型 ==============
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),
            nn.Tanh(),                              # 隐藏层用 Tanh 拟合平滑曲线
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)

model = MLP()
print("模型结构:")
print(model)

# ============== 3. 损失函数 + 优化器 ==============
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# 学习率衰减：每 200 epoch 衰减为原来的一半，后期步长变小避免抖动
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)

# ============== 4. 训练循环 ==============
losses = []
epochs = 1000
for epoch in range(epochs):
    # 前向传播
    pred = model(x_train)
    loss = loss_fn(pred, y_train)

    # 反向传播 + 参数更新（下一篇详解）
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    scheduler.step()  # 每个 epoch 结束后更新学习率

    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch+1:4d} | loss = {loss.item():.6f} | lr = {scheduler.get_last_lr()[0]:.2e}")

# ============== 5. 可视化结果 ==============
with torch.no_grad():
    y_pred = model(x_train)

plt.figure(figsize=(10, 4))

# 左图：拟合效果
plt.subplot(1, 2, 1)
plt.scatter(x_train.numpy(), y_train.numpy(), s=10, alpha=0.5, label="True (y = x^2)")
plt.plot(x_train.numpy(), y_pred.numpy(), 'r-', lw=2, label="MLP prediction")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("MLP 拟合 y = x^2")
plt.grid(True, alpha=0.3)

# 右图：loss 曲线
plt.subplot(1, 2, 2)
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss")
plt.yscale("log")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("fit_x2.png", dpi=120, bbox_inches="tight")
print("已保存 fit_x2.png")