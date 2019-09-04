import torch
import torch.nn as nn
import torch.nn.functional as F


class PositiveCosineLoss(nn.Module):
  ''' Cosine loss with margin for the positive way'''
  def __init__(self, margin=0):
    super().__init__()
    self.margin = margin
    self.metric = nn.CosineSimilarity()


  def forward(self, x1, x2):
    sim = self.metric(x1, x2)
    loss = 1 - sim - self.margin
    loss = torch.clamp(loss, 0, 2) # TODO: Check max value
    # TODO: Do nonzero() before mean?
    return loss.mean()


class ZeroCosineLoss(nn.Module):
  ''' Loss for cosine where we push everything against the 0 line. Margin takes in both positive and negative direction around 0 '''
  def __init__(self, margin=0):
    super().__init__()
    self.margin = margin
    self.metric = nn.CosineSimilarity()


  def forward(self, x1, x2):
    sim = self.metric(x1, x2)
    # loss = torch.abs(sim - self.margin)
    loss = sim.abs()
    loss = loss - self.margin
    loss = torch.clamp(loss, 0, 1)
    return loss.mean()


if __name__ == '__main__':
  # loss_fn = PositiveCosineLoss(margin=0.5)
  loss_fn = ZeroCosineLoss(margin=0)
  x1 = torch.randn(4, 3)
  x2 = torch.randn(4, 3)
  
  loss = loss_fn(x1, x2)
  print(loss)