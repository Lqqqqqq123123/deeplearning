import torch
import torch.nn as nn

linear = nn.Linear(in_features=3, out_features=5, bias=True)
# 可训练的参数
print(linear.weight.shape, linear.bias.shape)


