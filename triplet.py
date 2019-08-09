import numpy as np
import torch

# TODO: How many positive example and how many negatives ratio?

def create_triplets(originals, transformed):
  batch_size = originals.size(0)
  n_repeat = (batch_size - 1) * 2

  anchors = originals.repeat_interleave(n_repeat, dim=0)
  positives = transformed.repeat_interleave(n_repeat, dim=0)

  mask = [i for i in range(batch_size**2) if i%(batch_size+1) != 0]
  n1 = transformed.repeat(n_repeat + 1, 1)[mask]
  n2 = originals.repeat(n_repeat + 1, 1)[mask]
  negatives = torch.stack((n1, n2), dim=1).view(-1, anchors.size(1))

  return anchors, positives, negatives

def create_doublets(embeddings):
  batch_size = embeddings.size(0)
  n_repeat = batch_size - 1

  anchors = embeddings.repeat_interleave(n_repeat, dim=0)
  negatives = embeddings.repeat(n_repeat + 1, 1)

  mask = [i for i in range(batch_size**2) if i%(batch_size+1) != 0]
  negatives = negatives[mask]

  doublet = torch.cat((anchors, negatives), dim=1)
  return doublet


def test():
  b_size = 8
  batch = range(1, b_size+1)
  batch2 = [b+.5 for b in batch]

  a = torch.tensor(batch)
  n_repeat = a.size(0) - 1
  a = a.repeat_interleave(n_repeat)
  # print(a)

  p = torch.tensor(batch2)
  p = p.repeat_interleave(n_repeat)
  # print(p)

  len_b = len(batch)
  n = torch.tensor(batch) # For doing originals -> N
  n = torch.tensor(batch2) # For doing transformed -> N
  n = n.repeat(len_b)
  # print(n)

  mask = [i for i in range(len_b**2) if i%(len_b+1) != 0]
  # print(mask)

  n = n[mask]
  # print(n)

  print("a: {},  p: {},  n: {}".format(a, p, n))
  print('n={} -> {} comparisons'.format(len_b, len(n)))


def test2():
  batch = [[1, 2], [3, 4]]
  n_repeat = len(batch)
  a = torch.tensor(batch)
  n = a.repeat_interleave(n_repeat, dim=1)
  p = a.repeat(n_repeat, 1)
  print(n)
  print(p)

def test3():
  b_size = 8
  batch = range(1, b_size+1)
  batch2 = [b+.5 for b in batch]

  a = torch.tensor(batch)
  n_repeat = (a.size(0) - 1) * 2
  a = a.repeat_interleave(n_repeat)
  # print(a)

  p = torch.tensor(batch2)
  p = p.repeat_interleave(n_repeat)
  # print(p)

  len_b = len(batch)
  n1 = torch.tensor(batch, dtype=torch.float) # For doing originals -> N
  n2 = torch.tensor(batch2) # For doing transformed -> N
  n1 = n1.repeat(len_b)
  n2 = n2.repeat(len_b)
  # print(n1)
  # print(n2)

  mask = [i for i in range(len_b**2) if i%(len_b+1) != 0]
  # print(mask)

  n1 = n1[mask]
  n2 = n2[mask]

  n = torch.stack((n1, n2), dim=1).view(1, -1)

  print("a: {},  p: {},  n: {}".format(a, p, n))
  print('n={} -> {} comparisons'.format(len_b, len(a)))

if __name__ == '__main__':
  test()
  # test2()
  # test3()
