import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F

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
class DistanceNet(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    n_features = config.n_model_features * 2
    self.fc1 = nn.Linear(n_features, n_features)
    self.fc2 = nn.Linear(n_features, n_features)
    self.fc3 = nn.Linear(n_features, 1)

  def forward(self, x):
    # ims_embeddings -> distances
    x = F.relu(self.fc1(x), inplace=True)
    x = F.sigmoid(self.fc3(x))
    return x

