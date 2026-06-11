import torch
import torch.nn as nn


linear = nn.Linear(2,3)
# 初始化
# 1. 常数初始化
nn.init.constant_(linear.weight, 1)
print(linear.weight, linear.weight.shape)

# 2. 随机初始化
nn.init.normal_(linear.weight, mean=0, std=1)
print(linear.weight, linear.weight.shape)

# 3. Xavier 初始化
nn.init.xavier_uniform_(linear.weight)
print(linear.weight, linear.weight.shape)