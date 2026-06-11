import torch
import torch.nn as nn

class MyLinear(nn.Module):
    def __init__(self, in_f, out_f, bias=True):
        super().__init__()
        # 核心：用 nn.Parameter 包装张量，并初始化
        self.weight = nn.Parameter(torch.randn(out_f, in_f))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_f))
        else:
            self.bias = None


    def forward(self, x):
        temp = x @ self.weight.T
        if self.bias is not None:
            return temp + self.bias
        return temp


if __name__ == '__main__':
    x = torch.randn(2, 3)
    linear = MyLinear(3, 5)

    y = linear(x)
    print(y)
