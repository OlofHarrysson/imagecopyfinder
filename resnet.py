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

    anchors, positives, negatives = create_triplets(original_emb, transf_emb)
    a_2_p = torch.cat((anchors, positives), dim=1) # Dist -> 0
    a_2_n = torch.cat((anchors, negatives), dim=1) # Dist -> 1

    distance_input = torch.cat((a_2_p, a_2_n))
    distance_output = self.distance_measurer(distance_input)
    return distance_output, anchors, positives, negatives

  def predict_embedding(self, inputs):
    with torch.no_grad():
      embeddings = self.feature_extractor(inputs)
      return embeddings

  def calc_distance(self, query_emb, database_emb):
    inputs = torch.cat((query_emb, database_emb), dim=1)
    with torch.no_grad():
      outputs = self.distance_measurer(inputs)

    return outputs.cpu()

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

# TODO: Make several distance metrics. One for ecluedian distance, one for cosine similarity, etc. Name it similarity measurer or something
class DistanceMeasurer():
  def __init__(self, config):
    d1 = nn.CosineSimilarity()
    d2 = nn.PairwiseDistance(p=1)
    self.distance_metrics = [d1, d2]

  def calc_similarities(self, query_emb, database_emb):
    similarities = {}
    for metric in self.distance_metrics:
      similarity = self._calc_similaritiy(query_emb, database_emb, metric)
      print(metric)
      qwe

  def _calc_similarity(self, query_emb, database_emb, metric):
    distances = torch.tensor([])
    for db_entries, db_emb in database.items():
      dd = metric(query, db_emb)
      distances = torch.cat((distances, dd))
      
    return distances