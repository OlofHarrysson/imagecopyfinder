from itertools import combinations
import torch
import torch.nn as nn

for i in range(33):
  inp = 'a' * i
  ans = list(combinations(inp, 2))
  # print(ans)
  # print(len(ans))
  print(f'i={i} -> {len(ans)}')

