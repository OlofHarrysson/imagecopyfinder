import numpy as np
import torch

# TODO: How many positive example and how many negatives ratio?

def create_triplets(originals, transformed):
  batch_size = originals.size(0)
  n_repeat = batch_size - 1

  anchors = originals.repeat_interleave(n_repeat, dim=0)
  positives = transformed.repeat_interleave(n_repeat, dim=0)
  negatives = transformed.repeat(n_repeat + 1, 1)
  # negatives = originals.repeat(n_repeat + 1, 1)

  mask = [i for i in range(batch_size**2) if i%(batch_size+1) != 0]
  negatives = negatives[mask]

  # TODO: Negatives are not based on transform. Error?
  # asdas

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
  b_size = 3
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
  # n = torch.tensor(batch) # For doing originals -> N
  n = torch.tensor(batch2) # For doing transformed -> N
  n = n.repeat(len_b)

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

if __name__ == '__main__':
  test()
  # test2()
