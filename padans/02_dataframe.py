import pandas as pd
import numpy as np

# 创建 df
df1 = pd.DataFrame(
    {
        "A": [1, 2, 3],
        "B": [4, 5, 6],
    },
    columns=["B", "A"],
    index=['a', 'b', 'c']
)

print(df1)
print(df1.iloc[:, :])
print(df1.loc['a', ['A', 'B']])

