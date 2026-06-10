from torch.utils.data import DataLoader, Dataset,TensorDataset
import torch.nn as nn
import torch

x = torch.randn(2, 3)
y = torch.tensor([1, 0])

dataset = TensorDataset(x, y)

dataloader = DataLoader(
    dataset=dataset,
    batch_size=1,
    shuffle=True,
)

for x_batch, y_batch in dataloader:
    print(x_batch, y_batch)