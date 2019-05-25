import torchvision.models as models
import torch.nn as nn

class Resnet18(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.device = 'cuda' if config.use_gpu else 'cpu'
    self.basenet = models.resnet18(pretrained=True)
    self.basenet.fc = nn.Linear(512, config.n_model_features)
    
  def forward(self, x):
    x = x.to(self.device)
    return self.basenet(x)



class DistanceNet(nn.Module):
  def __init__(self, config):
    n_features = config.n_model_features
    self.fc1 = nn.Linear(n_features, n_features)
    self.fc2 = nn.Linear(n_features, n_features)
    self.fc3 = nn.Linear(n_features, 1)

  def forward(self, x):
    print('borokk')


