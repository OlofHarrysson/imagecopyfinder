import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
from triplet import create_triplets, create_doublets
from collections import defaultdict
from .resnet import *
from .resnet import BasicBlock
from .pooling import *
from pathlib import Path
from .metrics import DistanceMeasurer

class DistanceNet(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.device = 'cuda' if config.use_gpu else 'cpu'
    self.similarity_net = SimilarityNet(config)
    self.feature_extractor = FeatureExtractor(config, self.device)
    self.distance_measurer = DistanceMeasurer(config, self.feature_extractor, self.similarity_net)

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

  def save(self, path):
    save_dir = Path(path).parent
    save_dir.mkdir(exist_ok=True, parents=True)
    print("Saving Weights @ " + path)
    torch.save(self.state_dict(), path)

  def load(self, path):
    print('Loading weights from {}'.format(path))
    weights = torch.load(path, map_location='cpu')
    self.load_state_dict(weights, strict=False)

class FeatureExtractor(nn.Module):
  def __init__(self, config, device):
    super().__init__()
    self.basenet = resnet18(pretrained=config.pretrained)
    # self.basenet = resnet50(pretrained=config.pretrained)
    # for param in self.basenet.parameters():
    #   param.requires_grad = False

    n_features = self.basenet.fc.in_features
    self.pool = nn.AdaptiveAvgPool2d((1, 1))
    # self.pool = nn.AdaptiveMaxPool2d((1, 1))
    # self.pool = GeneralizedMeanPoolingManyP(n_features)

    # self.sim_weights = nn.Parameter(torch.ones(1, n_features))
    # exp = torch.tensor(np.exp(-np.linspace(0, 0.5, num=n_features))).float()
    # self.sim_weights = nn.Parameter(exp)

    self.fc = nn.Linear(n_features, config.n_model_features)
    # self.pool = AvgMaxPool()
    # self.fc = nn.Linear(n_features*5, config.n_model_features)

  def forward(self, x):
    x = self.basenet(x)
    x = self.pool(x)
    x = x.reshape(x.size(0), -1)
    x = self.fc(x)
    # x = F.normalize(x, p=2)
    return x

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