import torch
import torch.nn as nn

class myModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 3)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(3,1)

        # 初始化网络参数
        nn.init.xavier_normal(self.fc1.weight)
        nn.init.kaiming_normal(self.fc2.weight)

    def forward(self, x):
        # 中间结果1 (n, 2) -> (n, 3)
        m1 = self.relu(self.fc1(x))
        # 最终结果 (n,3) -> (n,1)
        return self.fc2(m1)

if __name__ == '__main__':
    t = torch.randn(32, 2)
    model = myModule()
    print(model(t))
    print(model.__dict__)

