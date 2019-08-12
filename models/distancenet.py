import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
from triplet import create_triplets, create_doublets
from collections import defaultdict
from .resnet import *
from .resnet import BasicBlock
from .pooling import *

class DistanceNet(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.device = 'cuda' if config.use_gpu else 'cpu'
    self.feature_extractor = FeatureExtractor(config)
    # self.similarity_net = SimilarityNet(config)
    self.distance_measurer = DistanceMeasurer(config)

  def forward(self, inputs):
    inputs = inputs.to(self.device)
    embeddings = self.feature_extractor(inputs)
    original_emb, transf_emb = embeddings.chunk(2)

    return original_emb, transf_emb

  def predict_embedding(self, inputs):
    inputs = inputs.to(self.device)
    with torch.no_grad():
      embeddings = self.feature_extractor(inputs)
      return embeddings

  def similarities(self, query_emb, database):
    return self.distance_measurer.calc_similarities(query_emb, database)

  def corrects(self, query_embs, database_embs):
    return self.distance_measurer.corrects(query_embs, database_embs)

  def cc_similarity_net(self, anchors, positives, negatives):
    a_p = torch.cat((anchors, positives), dim=1)
    a_n = torch.cat((anchors, negatives), dim=1)
    a_p_out = self.similarity_net(a_p)
    a_n_out = self.similarity_net(a_n)
    return a_p_out, a_n_out

class FeatureExtractor(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.basenet = resnet18(pretrained=config.pretrained)
    # self.basenet = resnet50(pretrained=config.pretrained)
    # for param in self.basenet.parameters():
    #   param.requires_grad = False

    n_features = self.basenet.fc.in_features
    # self.pool = nn.AdaptiveAvgPool2d((1, 1))
    # self.fc = nn.Linear(n_features, config.n_model_features)
    self.pool = AvgMaxPool()
    # self.pool = GeneralizedMeanPoolingP()
    # self.pool = GeneralizedMeanPoolingManyP(n_features, norm=1)
    self.fc = nn.Linear(n_features*5, config.n_model_features)


  def forward(self, x):
    x = self.basenet(x)
    x = self.pool(x)
    x = x.reshape(x.size(0), -1)
    x = self.fc(x)
    return x

  def get_params(self):
    return 

class SimilarityNet(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.device = 'cuda' if config.use_gpu else 'cpu'
    n_features = config.n_model_features
    self.fc1 = nn.Linear(2 * n_features, n_features)
    self.fc2 = nn.Linear(n_features, n_features)
    self.fc3 = nn.Linear(n_features, n_features)
    self.end = nn.Linear(n_features, 1)

  def forward(self, inputs):
    x = inputs.to(self.device)
    x = F.leaky_relu(self.fc1(x))
    x = F.leaky_relu(self.fc2(x))
    x = F.leaky_relu(self.fc3(x))
    return torch.sigmoid(self.end(x))

class DistanceMeasurer():
  def __init__(self, config):
    self.distance_metrics = []
    add_metric = lambda m: self.distance_metrics.append(m)
    add_metric(CosineSimilarity())
    # add_metric(EuclidianDistance1Norm())
    # add_metric(EuclidianDistance2Norm())
    # add_metric(EuclidianDistanceTopX(config.top_x))
    # add_metric(SimNet(sim_net))

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










