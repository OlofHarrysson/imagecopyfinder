import numpy as np
import torch

import torch.nn as nn
from torch.nn.modules import Module
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import math

class GeneralizedMeanPooling(Module):
    """Applies a 2D power-average adaptive pooling over an input signal composed of several input planes.
    The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
        - At p = infinity, one gets Max Pooling
        - At p = 1, one gets Average Pooling
    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.
    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.
    """

    def __init__(self, norm, output_size=1, eps=1e-6):
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        return F.adaptive_avg_pool2d(x, self.output_size).pow(1. / self.p)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + str(self.p) + ', ' \
            + 'output_size=' + str(self.output_size) + ')'


class GeneralizedMeanPoolingP(GeneralizedMeanPooling):
    """ Same, but norm is trainable
    """
    def __init__(self, norm=3, output_size=1, eps=1e-6):
        super(GeneralizedMeanPoolingP, self).__init__(norm, output_size, eps)
        self.p = Parameter(torch.ones(1) * norm)


class GeneralizedMeanPoolingManyP(GeneralizedMeanPooling):
  """ One p for each filter
  """
  def __init__(self, n_filters, norm=3, output_size=1, eps=1e-6):
    super().__init__(norm, output_size, eps)
    self.p = Parameter(torch.ones(n_filters) * norm)

  def forward(self, x):
    p = self.p.repeat(x.size(0), 1).unsqueeze(-1).unsqueeze(-1)
    x = x.clamp(min=self.eps).pow(p)
    return F.adaptive_avg_pool2d(x, self.output_size).pow(1. / p)


class AvgMaxPool(Module):
  # TODO: We can make it so that we have several output sizes. 1 == filter makes one activations per map, 2,2 makes it 4, etc.
  def __init__(self, output_size=1):
    super().__init__()
    self.output_size = output_size
    self.mean_pools = [GeneralizedMeanPooling(i) for i in range(2, 5)]

  def forward(self, x):
    x1 = F.adaptive_avg_pool2d(x, self.output_size)
    x2 = F.adaptive_max_pool2d(x, self.output_size)
    x3 = torch.cat((x1, x2), dim=1)

    for pool in self.mean_pools:
      x3 = torch.cat((x3, pool(x)), dim=1)

    return x3