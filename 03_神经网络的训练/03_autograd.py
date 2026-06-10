import torch
import torch.nn as nn

# 创建张量，并跟踪 x 构建计算图
# x = torch.tensor(2.0, requires_grad=True)
#
# # 计算 y
# y = x ** 2 + 3 * x + 1
#
# y.backward()
#
# print(x.grad.item())

# 非标量求导
x = torch.tensor([1, 2, 3], requires_grad=True, dtype=torch.float32)

y = x ** 2

y.backward(gradient=torch.ones_like(y))
print(x.grad) 


