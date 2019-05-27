import visdom
import torch
import torch.nn as nn
import numpy as np

def clear_envs(viz):
  [viz.close(env=env) for env in viz.get_env_list()] # Kills wind
  # [viz.delete_env(env) for env in viz.get_env_list()] # Kills envs

class Logger():
  def __init__(self, config):
    self.config = config
    self.viz = visdom.Visdom(port='6006')
    clear_envs(self.viz)

  def easy_or_hard(self, anchors, positives, negatives, margin, step):
    # TODO: Change? Still relavant for the triplet loss
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

  def log_distance_accuracy(self, positives, negatives, step):
    is_better = positives < negatives
    accuracy = is_better.sum().item() / is_better.numel()

    self.viz.line(
      Y=[accuracy],
      X=[step],
      update='append',
      win='DistAccuracy',
      opts=dict(
          xlabel='Steps',
          ylabel='Accuracy',
          title='Distance Accuracy',
          ytickmin = 0,
          ytickmax = 1
      )
    )

  def log_accuracy(self, ranks, step):
    n_ranks = len(ranks)
    top_x = lambda t_x: len([i for i in ranks if i <= t_x]) / n_ranks
    top_1, top_3, top_5 = top_x(1), top_x(3), top_x(5)

    Y = np.array([top_1, top_3, top_5]).reshape((1, 3))
    self.viz.line(
      Y=Y,
      X=[step],
      update='append',
      win='Accuracy',
      opts=dict(
          xlabel='Steps',
          ylabel='Validation Accuracy',
          title='Top-k Validation Accuracy',
          ytickmin = 0,
          ytickmax = 1,
          legend=['Top1', 'Top3', 'Top5'],
      )
    )

  def log_rank(self, ranks, step):
    # TODO: Output some kind of mean for the distro?

    # Bins several values together when there are a lot of ranks
    self.viz.histogram(
      X=[ranks],
      win='Rank',
      opts=dict(
          xlabel='Rank',
          ylabel='Number of Predictions',
          title='~Validation Rank',
      )
    )

  def log_loss(self, pos_loss, neg_loss, triplet_loss, loss, step):
    Y = torch.Tensor([loss, triplet_loss, pos_loss, neg_loss]).numpy()
    self.viz.line(
      Y=Y.reshape((1,4)),
      X=[step],
      update='append',
      win='TotalLoss',
      opts=dict(
          xlabel='Steps',
          ylabel='Loss',
          title='Training Loss',
          legend=['Total', 'Triplet', 'Positive', 'Negative']

      )
    )

  def log_lr(self, lr, step):
    self.viz.line(
      Y=[lr],
      X=[step],
      update='append',
      win='learning_rate',
      opts=dict(
          xlabel='Steps',
          ylabel='Accuracy',
          title='Learning rate',
      )
    )