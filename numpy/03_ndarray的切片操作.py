import numpy as np

arr = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

# 索引操作
print(arr[0])
print(arr[0][-1])

# 切片 st:ed:gap,
print(arr[1:, 1:])

# 花式索引
arr1 = np.array([10, 20, 30, 40, 50])
print(arr1[[0, 2, 4]])
print(arr1[arr1 > 30])
print(arr1[[True, False, True, False, True]])


