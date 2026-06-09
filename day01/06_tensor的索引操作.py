import torch

t1 = torch.tensor([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

print(t1.shape, t1.ndim)

# 普通索引
print(t1[0, 0]) # 1
# 范围索引（切片）
print(t1[:2, :2])

# 列表索引（花式索引）
print(t1[[0, 1, 2], [0, 1, 2]])

# 列表嵌套索引
print(t1[[0, 1, 2], 0])

print(t1 > 5)

print(t1[t1 > 5])