import torch

t1 = torch.tensor(
    [
        [
            [1, 2, 3],
            [4, 5, 6]
        ],
        [
            [1, 2, 3],
            [4, 5, 6]
        ]
    ]
)
print(t1.shape)
t2 = t1.sum(dim=0)

print(t2)
