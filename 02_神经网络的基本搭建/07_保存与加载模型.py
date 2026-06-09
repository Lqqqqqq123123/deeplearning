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
# 保存模型
# torch.save(net.state_dict(), 'model.pth')
# 加载模型
net.load_state_dict(torch.load('model/model.pth'))