import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
from triplet import create_triplets, create_doublets

class DistanceNet(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.device = 'cuda' if config.use_gpu else 'cpu'
    self.feature_extractor = Resnet18(config)
    self.distance_measurer = DistanceMeasurer(config)

  def forward(self, inputs):
    inputs = inputs.to(self.device)
    embeddings = self.feature_extractor(inputs)
    original_emb, transf_emb = embeddings.chunk(2)

    return create_triplets(original_emb, transf_emb)

  def predict_embedding(self, inputs):
    with torch.no_grad():
      embeddings = self.feature_extractor(inputs)
      return embeddings

  def similarities(self, query_emb, database):
    return self.distance_measurer.calc_similarities(query_emb, database)

class Resnet18(nn.Module):
  def __init__(self, config):
    super().__init__()
    n_features = config.n_model_features
    self.basenet = models.resnet18(pretrained=config.pretrained)
    # self.basenet = models.resnet34(pretrained=config.pretrained)
    self.basenet.fc = nn.Linear(self.basenet.fc.in_features, n_features)
    self.fc1 = nn.Linear(n_features, n_features)
    self.fc2 = nn.Linear(n_features, n_features)

    # TODO: Readlines writelines snippet
    
  def forward(self, x):
    x = F.relu(self.basenet(x), inplace=True)
    x = F.relu(self.fc1(x), inplace=True)
    x = self.fc2(x)
    return x

class DistanceMeasurer():
  def __init__(self, config):
    fn1 = CosineSimilarity()
    fn2 = EuclidianDistance()
    self.distance_metrics = [fn1, fn2]

  def calc_similarities(self, query_emb, database):
    ''' Returns similarities, a dict with 1-dim tensors for query to all in database '''
    similarities = {}
    for metric in self.distance_metrics:
      similarity = self._calc_similarity(query_emb, database, metric)
      similarities[str(metric)] = similarity

    return similarities

  def _calc_similarity(self, query_emb, database, metric):
    similarities = []
    for db_entries, db_emb in database.items():
      similarity = metric(query_emb, db_emb)
      similarities.append(torch.tensor(similarity))

    return torch.stack(similarities)















def assert_range(func):
  def wrapper(*args, **kwargs):
    similarity = func(*args, **kwargs)

    assert type(similarity) == float, f'Similarity needs to be a float but was of type {type(similarity)}'

    min_val, max_val = 0, 1
    is_ok = similarity >= min_val and similarity <= max_val
    assert is_ok, f'Similarity needs to be in range {min_val} - {max_val}. Function {func} gave {similarity.item()} instead'

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
    return self.func(query_emb, db_emb).item()

class EuclidianDistance(SimilarityMetric):
  def __init__(self):
    self.func = nn.PairwiseDistance(p=1)

  @assert_range
  def __call__(self, query_emb, db_emb):
    distance = self.func(query_emb, db_emb)
    return 1 - F.tanh(distance).item()