import torch
from torch import nn

# 形状 (5, 3)：5 个样本，每个 3 个特征
x = torch.randint(0, 10, (5, 3)).float()
print(x)

# 定义 BN 层：num_features = 特征数 C
bn = nn.BatchNorm1d(num_features=3)
y = bn(x)
print(y)

# l2 正则化
optimizer = torch.optim.AdamW([bn.weight],  lr=0.01, weight_decay=0.01)

print(bn.weight, bn.weight.size(), bn.bias, bn.bias.size())


# dropout
dropout = nn.Dropout(p = 0.5)

res = dropout(x)
print(res)