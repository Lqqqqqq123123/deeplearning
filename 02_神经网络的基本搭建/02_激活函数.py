import torch
import torch.nn as nn

# sigmoid
x = torch.tensor([0.7, -1.2, 2.5, 0, -3.1])

# sigmoid
sigmoid = nn.Sigmoid()
x1 = sigmoid(x)
print(x1)

x2 = torch.sigmoid(x)
print(x2)

x3 = x.sigmoid()
print(x3)


# tanh
x4 = torch.tanh(x)
print(x4)

x5 = x.tanh()
print(x5)

tanh = nn.Tanh()

x6 = tanh(x)
print(x6)

# softmax
x7 = torch.softmax(x, dim=-1)
print(x7)
res = 0
for i in x7:
    res += i.item()
    print(res)