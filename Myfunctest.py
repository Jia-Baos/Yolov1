import numpy as np
import torch

r_grid = np.array(list(range(14)))
# 对数字进行重复，repeats：次数，axis：重复的维度
r_grid = np.repeat(r_grid, repeats=28, axis=0)  # [0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 ...]
print(np.size(r_grid))
c_grid = np.array(list(range(14)))
# [np.newaxis, :]，扩充一个维度
c_grid = np.repeat(c_grid, repeats=2, axis=0)[np.newaxis, :]    # [[0 0 1 1 2 2 3 3 4 4 5 5 6 6]]
c_grid = np.repeat(c_grid, repeats=14, axis=0).reshape(-1)  # [0 0 1 1 2 2 3 3 4 4 5 5 6 6 0 0 1 1 2 2 3 3 4 4 5 5 6 6...]
print(np.size(c_grid))

input = torch.randint(low=0, high=10, size=(10, 1))
print(input)
input = torch.clamp(input, min=2)
print(input)
