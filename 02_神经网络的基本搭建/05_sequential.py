import torch
import torch.nn as nn

net = nn.Sequential(
    nn.Linear(3, 4),
    nn.Tanh(),
    nn.Linear(4, 4),
    nn.ReLU(),
    nn.Linear(4, 2),
    nn.Sigmoid()
)

x = torch.randn(10, 3)
y = net(x)
print(y)