import torch

t1 = torch.randn(2, 3, 4)
# 这个要被弃用了
# print(t1.T.shape)
t2 = t1.permute(list(range(t1.ndim - 1, -1, -1)))
print(t1.shape, t2.shape)

t3 = torch.arange(12)
print(t3)