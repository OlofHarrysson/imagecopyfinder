import visdom
import torch
import torch.nn as nn
import numpy as np
from utils.utils import EMAverage
from models.metrics import CosineSimilarity
import plotly.graph_objects as go


def clear_envs(viz):
  [viz.close(env=env) for env in viz.get_env_list()]  # Kills wind
  # [viz.delete_env(env) for env in viz.get_env_list()] # Kills envs


class Logger():
  def __init__(self, config):
    self.config = config
    self.viz = visdom.Visdom(port='6006')
    clear_envs(self.viz)

    self.corrects_average = None
    self.loss_percent_average = None

  def init_corrects_average(self, corrects):
    self.corrects_average = {}
    for key in corrects.keys():
      self.corrects_average[key] = EMAverage(30)

  def init_loss_percent_average(self, loss_dict):
    self.loss_percent_average = {}
    for key in loss_dict.keys():
      self.loss_percent_average[key] = EMAverage(30)

  def easy_or_hard(self, anchors, positives, negatives, margin, step):
    dist = nn.PairwiseDistance(p=self.config.distance_norm)
    a_to_p = dist(anchors, positives)
    a_to_n = dist(anchors, negatives)

    easy = a_to_p + margin < a_to_n
    hard = a_to_p > a_to_n
    semi_hard = torch.ones_like(easy) ^ (easy + hard)

    n_comp = easy.size(0)
    ee = easy.sum(dtype=torch.float32) / n_comp
    hh = hard.sum(dtype=torch.float32) / n_comp
    sh = semi_hard.sum(dtype=torch.float32) / n_comp

    Y = torch.Tensor([ee, ee + sh, ee + sh + hh]).numpy()
    self.viz.line(Y=Y.reshape((1, 3)),
                  X=[step],
                  update='append',
                  win='TripletDifficulty',
                  opts=dict(fillarea=True,
                            xlabel='Steps',
                            ylabel='Percentage',
                            title='Example Difficulty',
                            stackgroup='one',
                            legend=['Easy', 'Semi', 'Hard']))

    n_not_easy = hard.sum() + semi_hard.sum()
    self.viz.line(Y=[n_not_easy.item()],
                  X=[step],
                  update='append',
                  win='Examples_loss',
                  opts=dict(
                    fillarea=True,
                    xlabel='Steps',
                    ylabel='Number of Examples',
                    title='#Examples with Loss',
                  ))

  def log_accuracy(self, ranks, step, name):
    n_ranks = len(ranks)
    top_x = lambda t_x: len([i for i in ranks if i <= t_x]) / n_ranks
    top_1, top_2, top_3, top_5 = top_x(1), top_x(2), top_x(3), top_x(5)

    Y = np.array([top_1, top_2, top_3, top_5]).reshape((1, -1))
    self.viz.line(
      Y=Y,
      X=[step],
      update='append',
      win='Accuracy' + name,
      opts=dict(
        xlabel='Steps',
        ylabel='Accuracy',
        title=f'Val Accuracy {name}',
        # ytickmin = 0,
        # ytickmax = 1,
        legend=['Top1', 'Top2', 'Top3', 'Top5'],
      ))

  def log_rank(self, ranks, step, name):
    # TODO: Output some kind of mean for the distro?

    # Bins several values together when there are a lot of ranks
    self.viz.histogram(X=[ranks],
                       win='Rank' + name,
                       opts=dict(
                         xlabel='Rank',
                         ylabel='Number of Predictions',
                         title=f'~Validation Rank {name}',
                       ))

  def log_loss(self, loss, step):
    Y = torch.Tensor([loss]).numpy()
    self.viz.line(Y=Y.reshape((1, 1)),
                  X=[step],
                  update='append',
                  win='TotalLoss',
                  opts=dict(xlabel='Steps',
                            ylabel='Loss',
                            title='Training Loss',
                            legend=['Total']))

  def log_loss_percent(self, loss_dict, step):
    if self.loss_percent_average == None:
      self.init_loss_percent_average(loss_dict)

    legend, losses = [], []
    for name, loss in loss_dict.items():
      legend.append(name)
      avg_tracker = self.loss_percent_average[name]
      val = avg_tracker.update(loss.item())
      losses.append(val)

    tot_loss = sum(losses)
    temp_loss = 0
    Y = []
    for loss in losses:
      Y.append((temp_loss + loss) / tot_loss)
      temp_loss += loss

    self.viz.line(Y=np.array(Y).reshape(1, -1),
                  X=[step],
                  update='append',
                  win='losspercent',
                  opts=dict(fillarea=True,
                            xlabel='Steps',
                            ylabel='Percentage',
                            title='Loss Percentage',
                            stackgroup='one',
                            legend=legend))

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
    self.viz.line(Y=Y,
                  X=[step],
                  update='append',
                  win='Distance Accuracy',
                  opts=dict(
                    xlabel='Steps',
                    ylabel='Top-1 Accuracy',
                    title=f'Distance Accuracy',
                    ytickmin=0,
                    ytickmax=1,
                    legend=metrics,
                  ))

  def log_cosine(self, anchors, positives, negatives):
    metric = CosineSimilarity()

    pos_sim = metric(anchors, positives)
    pos_sim = pos_sim.cpu().detach().numpy()
    neg_sim = metric(anchors, negatives)
    neg_sim = neg_sim.cpu().detach().numpy()

    title_text = 'Cosine Distance'
    fig = go.Figure()

    violin_plot = lambda ys, side, name: go.Violin(
      y=ys,
      # box_visible=True,
      meanline_visible=True,
      spanmode='hard',
      side=side,
      x0='Pos/Neg',
      name=name,
    )

    fig.add_trace(violin_plot(pos_sim, 'negative', 'Positives'))
    fig.add_trace(violin_plot(neg_sim, 'positive', 'Negatives'))
    # TODO: Visdom doesn't work with title in layout

    fig.update_layout(shapes=[
      # Line Horizontal
      go.layout.Shape(
        type="line",
        x0=-0.5,
        y0=0.75,
        x1=0.5,
        y1=0.75,
        line=dict(
          width=2,
          dash="dot",
        ),
      ),
    ])

    self.viz.plotlyplot(fig, win=title_text)

  def log_violin(self, data, title):
    fig = go.Figure()

    violin_plot = lambda ys, name: go.Violin(
      y=ys,
      # box_visible=True,
      meanline_visible=True,
      spanmode='hard',
      x0='Mydata',
      name=name,
    )

    fig.add_trace(violin_plot(data, title))
    self.viz.plotlyplot(fig, win=title)

  def log_p(self, p, step):
    title_text = 'Generalized Mean Pooling'
    fig = go.Figure()

    p = p.detach().cpu().numpy()
    violin_plot = lambda ys, name: go.Violin(
      y=ys,
      # box_visible=True,
      meanline_visible=True,
      spanmode='hard',
      x0='Pos/Neg',
      name=name,
    )

    fig.add_trace(violin_plot(p, 'P-param'))
    self.viz.plotlyplot(fig, win=title_text)

  def log_image(self, image, caption):
    opts = dict(title=f'image_{caption}')
    self.viz.image(image, opts=opts)

  def log_weights(self, weights):
    title = 'weights'
    data = weights.cpu().squeeze().detach().numpy()
    fig = go.Figure()

    violin_plot = lambda ys, name: go.Violin(
      y=ys,
      # box_visible=True,
      meanline_visible=True,
      spanmode='hard',
      x0='Similarity Weights',
      name=name,
    )

    fig.add_trace(violin_plot(data, 'Similarity Weights'))
    self.viz.plotlyplot(fig, win=title)

    # Weights
    self.viz.line(Y=data,
                  X=np.array(range(len(data))),
                  win='line sim weights',
                  opts=dict(
                    xlabel='Steps',
                    ylabel='Percentage',
                    title=title,
                  ))

    # Sorted weights
    sorted_data, inds = torch.sort(weights, descending=True)
    self.viz.line(Y=sorted_data.cpu().squeeze().detach().numpy(),
                  X=np.array(range(len(data))),
                  win='line sim weights sorted',
                  opts=dict(
                    xlabel='Steps',
                    ylabel='Percentage',
                    title='Sorted Similarity Weights',
                  ))
