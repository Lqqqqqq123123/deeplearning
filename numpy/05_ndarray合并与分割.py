import numpy as np
# 合并
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(np.concatenate([a, b]))
print(np.vstack([a, b])) # 垂直堆叠
print(np.hstack([a, b])) # 水平堆叠
print(np.stack([a, b], axis=1)) # 新增维度堆叠

# 分割
arr = np.arange(12).reshape(3, 4)
np.split(arr, 3)                  # 分成 3 份
np.split(arr, [1, 3], axis=1)     # 在第 1、3 列分割
np.vsplit(arr, 3)                 # 垂直分割
np.hsplit(arr, 2)                 # 水平分割