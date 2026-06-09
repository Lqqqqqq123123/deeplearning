import torch
import numpy as np
# 按内容去创建
tensor1 = torch.tensor([1, 2, 3])
print(tensor1.ndim, tensor1.shape, tensor1.itemsize, tensor1.dtype)
tensor2 = torch.tensor(np.array([[1, 2, 3], [4, 5, 6]]))
print(tensor2.ndim, tensor2.shape, tensor2.itemsize, tensor2.dtype)

# 按形状去创建
tensor3 = torch.zeros(2, 3) # 创建一个 2 行 3 列的零矩阵
tensor4 = torch.ones(2, 3) # 创建一个 2 行 3 列的 1 矩阵

# 随机创建
tensor5 = torch.rand(4, 4)
tensor6 = torch.randint(0, 10, (2, 3)) # 创建一个 2 行 3 列的随机矩阵，范围是 0-10、
tensor7 = torch.randn(4, 4) # 创建一个 4 行 4 列的随机矩阵，范围是 -1-1（标准正态分布）

# 序列创建
tensor8 = torch.arange(0, 10)
tensor9 = torch.linspace(0, 10, 10)
tensor10 = torch.logspace(0, 10, 10)

print(tensor8, tensor9, tensor10)
