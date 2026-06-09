import torch

t = torch.tensor([1, 2, 3])
print(t.dtype)

nt = t.type(torch.float32)
nnt = t.to(torch.float32)
print(nt.dtype, nnt.dtype)