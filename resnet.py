import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
from triplet import create_triplets, create_doublets


class Resnet18(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.device = 'cuda' if config.use_gpu else 'cpu'

    self.basenet = models.resnet18(pretrained=True)
    self.basenet.fc = nn.Linear(512, config.n_model_features)
    
  def forward(self, x):
    x = x.to(self.device)
    return self.basenet(x)

# TODO: Instead of triplet training, could we have a classifier/discriminator try to tell us the distance between a-p, a-n? Should output 1=similar for a-p and 0 for a-n. This should be better right?
class DistanceMeasurer(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.device = 'cuda' if config.use_gpu else 'cpu'

    n_features = config.n_model_features * 2
    self.fc1 = nn.Linear(n_features, n_features)
    self.fc2 = nn.Linear(n_features, n_features)
    self.fc3 = nn.Linear(n_features, 1)

  def forward(self, x):
    x = x.to(self.device)
    x = F.relu(self.fc1(x), inplace=True)
    x = torch.sigmoid(self.fc3(x))
    return x


class DistanceNet(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.device = 'cuda' if config.use_gpu else 'cpu'
    self.feature_extractor = Resnet18(config)
    self.distance_measurer = DistanceMeasurer(config)

  def forward(self, inputs):
    embeddings = self.feature_extractor(inputs)
    original_emb, transf_emb = embeddings.chunk(2)

    anchors, positives, negatives = create_triplets(original_emb, transf_emb)
    a_2_p = torch.cat((anchors, positives), dim=1) # Dist -> 0
    a_2_n = torch.cat((anchors, negatives), dim=1) # Dist -> 1

    distance_input = torch.cat((a_2_p, a_2_n))
    distance_output = self.distance_measurer(distance_input)
    return distance_output

  def predict(self, x):
    pass