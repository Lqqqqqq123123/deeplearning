import torch
import torch.nn as nn

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 4)
        self.fc2 = nn.Linear(4, 4)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x



if __name__ == '__main__':
    model = MyNet()
    # 1. 直接拿到这个模型的子模块，然就获得当前权重
    print(model.fc1.weight.shape)
    # print(type(model.parameters()), type(model.named_parameters()))
    # 2. 拿到这个模型所有的参数
    for name, param in model.named_parameters():
        print(f'{name=}, {param.shape=}')
    # 3. state_dict
    for k, v in model.state_dict().items():
        print(f'{k=}, {v.shape=}')
    # torchsummary
    from torchsummary import summary
    summary(model, (3, ), batch_size=1, device='cpu')
    