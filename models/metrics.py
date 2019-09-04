import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
# from triplet import create_triplets, create_doublets
# from collections import defaultdict
import numpy as np


class DistanceMeasurer():
  def __init__(self, config, model):
    self.distance_metrics = []
    add_metric = lambda m: self.distance_metrics.append(m)
    add_metric(CosineSimilarity())
    # add_metric(EuclidianDistance1Norm())
    # add_metric(EuclidianDistance2Norm())
    # add_metric(EuclidianDistanceTopX(config.top_x))
    add_metric(Weighted(model))

  def calc_similarities(self, query_emb, database):
    ''' Returns similarities, a dict with 1-dim tensors for query to all in database '''
    database_embs = list(database.values())
    database_embs = torch.stack(database_embs).squeeze()

    similarities = {}
    for metric in self.distance_metrics:
      similarity = self._calc_similarity(query_emb, database_embs, metric)
      similarities[str(metric)] = similarity

    return similarities

  def _calc_similarity(self, query_emb, database_embs, metric):
    # TODO: Use expand_as if no training is to be done. Saves memory
    # query_embs = query_emb.expand_as(database_embs)
    query_embs = query_emb.repeat((database_embs.size(0), 1))
    return metric(query_embs, database_embs)


  def corrects(self, query_embs, database_embs):
    query_embs = query_embs.cpu()
    database_embs = database_embs.cpu()
    corrects = defaultdict(lambda: [])

    for q_ind, query in enumerate(query_embs):
      for metric in self.distance_metrics:
        sim = self._calc_similarity(query, database_embs, metric)

        _, max_ind = sim.max(0)
        corrects[str(metric)].append((max_ind == q_ind).item())

    # print(sim)
    return corrects


def assert_range(func):
  def wrapper(*args, **kwargs):
    similarity = func(*args, **kwargs)

    eps = 1e-4
    min_val, max_val = 0, 1
    is_ok = similarity.ge(min_val-eps).all() and similarity.le(max_val+eps).all()
    assert is_ok, f'Similarity needs to be in range {min_val} - {max_val}. Function {func} gave {similarity} instead'

    return similarity
  return wrapper

class SimilarityMetric:
  def __call__(self, query_emb, db_emb):
    raise NotImplementedError('Similarity metrics need to return a similarity in the range 0 - 1 where 1 is very similar')

  def __str__(self):
    return type(self).__name__

class CosineSimilarity(SimilarityMetric):
  def __init__(self):
    self.func = nn.CosineSimilarity()

  @assert_range
  def __call__(self, query_emb, db_emb):
    cos = self.func(query_emb, db_emb)
    return (cos + 1) / 2

class EuclidianDistance1Norm(SimilarityMetric):
  def __init__(self):
    self.func = nn.PairwiseDistance(p=1)

  @assert_range
  def __call__(self, query_emb, db_emb):
    distance = self.func(query_emb, db_emb)
    return 1 - torch.sigmoid(distance)

class EuclidianDistance2Norm(SimilarityMetric):
  def __init__(self):
    self.func = nn.PairwiseDistance(p=2)

  @assert_range
  def __call__(self, query_emb, db_emb):
    distance = self.func(query_emb, db_emb)
    # TODO: Remap function. [0.5 - 1] > [0 - 1]
    return 1 - torch.sigmoid(distance)


class EuclidianDistanceTopX(SimilarityMetric):
  def __init__(self, top_x):
    self.top_x = top_x

  @assert_range
  def __call__(self, query_emb, db_emb):
    distance = torch.abs(query_emb - db_emb)
    best_distances, _ = distance.topk(self.top_x, dim=1, largest=False)
    return 1 - torch.sigmoid(best_distances.sum(dim=1))


class SimNet(SimilarityMetric):
  def __init__(self, model):
    self.model = model

  @assert_range
  def __call__(self, query_emb, db_emb):
    embs = torch.cat((db_emb, query_emb), dim=1)
    self.model.eval()
    with torch.no_grad():
      outs = self.model(embs).squeeze()
    self.model.train()
    return outs


class Weighted(SimilarityMetric):
  def __init__(self, model):
    self.model = model
    self.func = nn.CosineSimilarity()

  @assert_range
  def __call__(self, query_emb, db_emb):
    diff = torch.abs(query_emb - db_emb)
    sort_vals, sort_inds = torch.sort(diff)
    
    sort_inds = sort_inds.view((1, -1))
    weights = self.model.sim_weights.view((1, -1)).cpu()

    s_weights = weights[:, sort_inds]
    s_weights = s_weights.view(query_emb.shape[0], -1)

    s_q = s_weights * query_emb
    s_db = s_weights * db_emb

    cos = self.func(s_q, s_db)
    return (cos + 1) / 2


if __name__ == '__main__':
  mm = Weighted(None)
  t1 = torch.randn(2, 4)
  t2 = torch.randn(2, 4)
  ans = mm(t1, t2)
  print(ans)

