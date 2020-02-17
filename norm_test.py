import torch
dist = torch.nn.PairwiseDistance()


def main():
  dim = 512
  # dim = int(1e5)
  loops = 1000
  min_dist, max_dist, avg_dist = check_dist(dim, loops)
  print('Dim=', dim, min_dist, max_dist, avg_dist)


def check_dist(dim, loops):
  min_dist, max_dist = 100000000, 0
  avg_dist = []
  for _ in range(loops):
    a = normed(dim)
    b = normed(dim)
    d = dist(a, b)
    if d < min_dist:
      min_dist = d
    if d > max_dist:
      max_dist = d

    avg_dist.append(d.item())

  avg_dist = sum(avg_dist) / len(avg_dist)
  return min_dist, max_dist, avg_dist


def normed(dim):
  x = torch.randn(1, dim)
  norm = x.norm(p=2, keepdim=True)
  x_normalized = x.div(norm)
  return x_normalized


if __name__ == '__main__':
  main()
