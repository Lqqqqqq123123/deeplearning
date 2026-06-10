import torch
import torch.nn as nn
from networkx.algorithms.centrality import trophic

target = torch.randn(5)
output = torch.randn(5)

# l1 loss，绝对误差
loss_l1 = nn.L1Loss()
res = loss_l1(output, target)
print(res)
# l2 loss，均方误差
loss_l2 = nn.MSELoss()
res = loss_l2(output, target)
print(res)

# 二元交叉熵损失
loss_bce = nn.BCELoss()
target = torch.tensor([0, 1, 0, 1, 0], dtype=torch.float32)
output = torch.randn(5)
res = loss_bce(torch.sigmoid(output),  target)
print(res)

# 多分类交叉熵损失
loss_ce = nn.CrossEntropyLoss()
target_origin = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
target = torch.zeros(5, 5)
target[[i for i in range(5)], target_origin] = 1 # one-hot 编码
print(target)
output = torch.randn(5, 5)
# 1. 必须先去 softmax 一下
res = loss_ce(torch.softmax(output, dim=-1), target) # 这里既可以传类别标签，也可以传 one-hot 编码


print(res)

