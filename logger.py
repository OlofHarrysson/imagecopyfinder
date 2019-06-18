import visdom
import torch
import torch.nn as nn
import numpy as np
from utils.utils import EMAverage

def clear_envs(viz):
  [viz.close(env=env) for env in viz.get_env_list()] # Kills wind
  # [viz.delete_env(env) for env in viz.get_env_list()] # Kills envs

class Logger():
  def __init__(self, config):
    self.config = config
    self.viz = visdom.Visdom(port='6006')
    clear_envs(self.viz)

    self.corrects_average = None

  def init_corrects_average(self, corrects):
    self.corrects_average = {}
    for key in corrects.keys():
      self.corrects_average[key] = EMAverage(30)

  def easy_or_hard(self, anchors, positives, negatives, margin, step):
    dist = nn.PairwiseDistance(p=self.config.distance_norm)
    a_to_p = dist(anchors, positives)
    a_to_n = dist(anchors, negatives)

    easy = a_to_p + margin < a_to_n
    hard = a_to_p > a_to_n
    semi_hard = torch.ones_like(easy) - (easy + hard)

    n_comp = easy.size(0)
    ee = easy.sum(dtype=torch.float32) / n_comp
    hh = hard.sum(dtype=torch.float32) / n_comp
    sh = semi_hard.sum(dtype=torch.float32) / n_comp

    Y = torch.Tensor([ee, ee+sh, ee+sh+hh]).numpy()
    self.viz.line(
      Y=Y.reshape((1, 3)),
      X=[step],
      update='append',
      win='TripletDifficulty',
      opts=dict(
          fillarea=True,
          xlabel='Steps',
          ylabel='Percentage',
          title='Example Difficulty',
          stackgroup='one',
          legend=['Easy', 'Semi', 'Hard']
      )
    )

    n_not_easy = hard.sum() + semi_hard.sum()
    self.viz.line(
      Y=[n_not_easy.item()],
      X=[step],
      update='append',
      win='Examples_loss',
      opts=dict(
          fillarea=True,
          xlabel='Steps',
          ylabel='Number of Examples',
          title='#Examples with Loss',
      )
    )

  def cosine_ez_hard(self, anchors, positives, negatives, margin, step):
    dist = nn.CosineSimilarity()
    a_to_p = dist(anchors, positives)
    a_to_n = dist(anchors, negatives)

    easy = a_to_p > a_to_n + margin
    hard = a_to_p < a_to_n
    semi_hard = torch.ones_like(easy) - (easy + hard)

    n_comp = easy.size(0)
    ee = easy.sum(dtype=torch.float32) / n_comp
    hh = hard.sum(dtype=torch.float32) / n_comp
    sh = semi_hard.sum(dtype=torch.float32) / n_comp

    Y = torch.Tensor([ee, ee+sh, ee+sh+hh]).numpy()
    self.viz.line(
      Y=Y.reshape((1, 3)),
      X=[step],
      update='append',
      win='TripletDifficulty',
      opts=dict(
          fillarea=True,
          xlabel='Steps',
          ylabel='Percentage',
          title='Example Difficulty',
          stackgroup='one',
          legend=['Easy', 'Semi', 'Hard']
      )
    )

    n_not_easy = hard.sum() + semi_hard.sum()
    self.viz.line(
      Y=[n_not_easy.item()],
      X=[step],
      update='append',
      win='Examples_loss',
      opts=dict(
          fillarea=True,
          xlabel='Steps',
          ylabel='Number of Examples',
          title='#Examples with Loss',
      )
    )

  def log_accuracy(self, ranks, step, name):
    n_ranks = len(ranks)
    top_x = lambda t_x: len([i for i in ranks if i <= t_x]) / n_ranks
    top_1, top_2, top_3, top_5 = top_x(1), top_x(2), top_x(3), top_x(5)

    Y = np.array([top_1, top_2, top_3, top_5]).reshape((1, -1))
    self.viz.line(
      Y=Y,
      X=[step],
      update='append',
      win='Accuracy'+name,
      opts=dict(
          xlabel='Steps',
          ylabel='Accuracy',
          title=f'Val Accuracy {name}',
          ytickmin = 0,
          ytickmax = 1,
          legend=['Top1', 'Top2', 'Top3', 'Top5'],
      )
    )

  def log_rank(self, ranks, step, name):
    # TODO: Output some kind of mean for the distro?

    # Bins several values together when there are a lot of ranks
    self.viz.histogram(
      X=[ranks],
      win='Rank'+name,
      opts=dict(
          xlabel='Rank',
          ylabel='Number of Predictions',
          title=f'~Validation Rank {name}',
      )
    )

  def log_loss(self, loss, step):
    Y = torch.Tensor([loss]).numpy()
    self.viz.line(
      Y=Y.reshape((1,1)),
      X=[step],
      update='append',
      win='TotalLoss',
      opts=dict(
          xlabel='Steps',
          ylabel='Loss',
          title='Training Loss',
          legend=['Total']

      )
    )

  def log_corrects(self, corrects, step):
    if self.corrects_average == None:
      self.init_corrects_average(corrects)

    metrics, accuracies = [], []
    for key, val in corrects.items():
      metrics.append(key)
      avg_tracker = self.corrects_average[key]
      acc = avg_tracker.update(sum(val) / len(val))
      accuracies.append(acc)

    Y = np.array(accuracies).reshape((1, -1))
    self.viz.line(
      Y=Y,
      X=[step],
      update='append',
      win='Distance Accuracy',
      opts=dict(
          xlabel='Steps',
          ylabel='Top-1 Accuracy',
          title=f'Distance Accuracy',
          ytickmin = 0,
          ytickmax = 1,
          legend=metrics,
      )
    )