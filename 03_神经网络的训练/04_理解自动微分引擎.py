import torch
import torch.nn as nn

# 1. 定义数据
x = torch.tensor([10.0])
y_true = torch.tensor([3.0])

# 2. 定义参数和偏置
w = torch.tensor([1.0], requires_grad=True)
b = torch.tensor([1.0], requires_grad=True)

# 3. 计算
z = w * x + b
loss_l2 = nn.MSELoss()

loss:torch.Tensor = loss_l2(z, y_true)

print(loss, loss.grad_fn.next_functions)
print(z, z.grad_fn)


# 4. 反向传播
loss.retain_grad()
z.retain_grad()

loss.backward()

print(loss.grad, z.grad)





