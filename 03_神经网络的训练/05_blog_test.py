import torch
import torch.nn as nn

t = torch.tensor([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

print(t[[0, 1, 2], 0])