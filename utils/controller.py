import torch, math, random
from utils.utils import ProgressbarWrapper as Progressbar
from data import TripletDataset
from torch.utils.data import DataLoader
from triplet import create_triplets, create_doublets
from logger import Logger
from utils.validator import Validator
from transform import Transformer, CropTransformer
from pathlib import Path
import torchvision.transforms as transforms
import numpy as np

# TODO: I can increase the augmentation depending on how many easy/hard there are. Possibly also increase margin?
# TODO: Change distance norm 2->1? I want the system to match because there are a lot of features that are close, not fuck up because one feature is bad and dominates the others. Could even select the top% features that matchest best.
# TODO: Could this be used as a pretraining system for triplet loss


# TODO: Can have a prepare data colab file that prepares the data and puts it in gdrive. It could download a dataset online. Then a user can upload its dataset to his/hers gdrive. Would be able to run the project from any computer without installation

# TODO: Now the anchor+positive is the same all the time. What if we expand the number of fakes for more combinations?


# TODO:
# More data


# TODO: After all conv2d layers, do adaptive pooling and continue with conv1d layers. At this point I don't care about spatial info anymore. Perhaps add a fc in after the adaptive pooling. But the reason is that conv layers use less parameters. Can try with fc layers as well.




def clear_output_dir():
  [p.unlink() for p in Path('output').iterdir()]


def init_training(model, config):
  torch.backends.cudnn.benchmark = True # Optimizes cudnn
  model.to(model.device) # model -> CPU/GPU

  # Optimizer & Scheduler
  # TODO CyclicLR
  params = filter(lambda p: p.requires_grad, model.parameters())
  optimizer = torch.optim.Adam(params, lr=config.start_lr)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.optim_steps/config.lr_step_frequency, eta_min=config.end_lr)

  return optimizer, scheduler


def train(model, config):
  # model_parameters = filter(lambda p: p.requires_grad, model.feature_extractor.parameters())
  # params = sum([np.prod(p.size()) for p in model_parameters])
  # print(params/1e6)
  # qwe

  # clear_output_dir()
  optimizer, lr_scheduler = init_training(model, config)
  logger = Logger(config)
  validator = Validator(model, logger, config)
  # transformer = Transformer()
  transformer = CropTransformer()
  margin = 5
  triplet_loss_fn = torch.nn.TripletMarginLoss(margin, p=config.distance_norm, swap=True)
  cos_loss_fn = torch.nn.CosineEmbeddingLoss(margin=0.1)
  similarity_loss_fn = torch.nn.BCELoss()

  # Data
  def collate(batch):
    im_sizes = config.image_input_size
    im_size = random.randint(im_sizes[0], im_sizes[1])
    uniform_size = transforms.Compose([
                       transforms.Resize((im_size, im_size)),
                       transforms.ToTensor()
    ])

    original_ims = [uniform_size(b[0]) for b in batch]
    transformed_ims = [uniform_size(b[1]) for b in batch]

    return torch.stack(original_ims), torch.stack(transformed_ims)

  batch_size = config.batch_size
  dataset = TripletDataset(config.dataset, transformer, config)
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=config.num_workers, collate_fn=collate)

  # Init progressbar
  n_batches = len(dataloader)
  n_epochs = math.ceil(config.optim_steps / n_batches)
  pbar = Progressbar(n_epochs, n_batches)

  optim_steps = 0
  val_freq = config.validation_freq

  # Training loop
  for epoch in pbar(range(1, n_epochs + 1)):
    for batch_i, data in enumerate(dataloader, 1):
      pbar.update(epoch, batch_i)

      # Validation
      # if optim_steps % val_freq == 0:
      #   validator.validate(optim_steps)

      # Decrease learning rate
      if optim_steps % config.lr_step_frequency == 0:
        lr_scheduler.step()

      optimizer.zero_grad()
      original, transformed = data

      inputs = torch.cat((original, transformed))

      outputs = model(inputs)
      original_emb, transf_emb = outputs
      anchors, positives, negatives = create_triplets(original_emb, transf_emb)
      
      # Triplet loss
      triplet_loss = triplet_loss_fn(anchors, positives, negatives)

      # Direct net loss
      # a_p, a_n = model.cc_similarity_net(anchors, positives, negatives)
      # net_match_loss = similarity_loss_fn(a_p, torch.ones_like(a_p))
      # net_not_match_loss = similarity_loss_fn(a_n, torch.zeros_like(a_n))
      # net_loss = net_match_loss + net_not_match_loss

      # Cosine similarity loss
      y_size = anchors.size(0)
      y = torch.ones(y_size).to(model.device)
      cos_match_loss = cos_loss_fn(anchors, positives, y)
      cos_not_match_loss = cos_loss_fn(anchors, negatives, -1 * y)
      cos_loss = cos_match_loss + cos_not_match_loss
      cos_loss = cos_not_match_loss

      # loss_dict = dict(triplet=triplet_loss, cos=cos_loss, net=net_loss)
      loss_dict = dict(triplet=triplet_loss, cos=cos_loss)
      # loss_dict = dict(cos=cos_loss)
      # loss_dict = dict(net=net_loss)
      # loss_dict = dict(triplet=triplet_loss)

      loss = sum(loss_dict.values())
      loss.backward()
      plot_grad_flow(logger, model.named_parameters())

      qweqw

      optimizer.step()
      optim_steps += 1

      corrects = model.corrects(transf_emb, original_emb)
      logger.easy_or_hard(anchors, positives, negatives, margin, optim_steps)
      logger.log_loss(loss, optim_steps)
      logger.log_loss_percent(loss_dict, optim_steps)
      logger.log_corrects(corrects, optim_steps)
      
      # Frees up GPU memory
      del data; del outputs

def plot_grad_flow(logger, named_parameters):
  import matplotlib.pyplot as plt
  import plotly.plotly as py
  import plotly.graph_objs as go

  ave_grads = []
  layers = []
  n_none, n_grad = 0, 0
  for n, p in named_parameters:
    if n == 'feature_extractor.basenet.fc.weight':
      continue # The resnet weights we dont use

    if n == 'feature_extractor.fc.weight':
      print(n)
      print(p.grad.shape)
      print(p)
      data = p.detach().numpy()
      data = data.flatten()
      print(data.shape)

      fig = {
          "data": [{
              "type": 'violin',
              "y": data,
              "box": {
                  "visible": True
              },
              "line": {
                  "color": 'black'
              },
              "meanline": {
                  "visible": True
              },
              "fillcolor": '#8dd3c7',
              "opacity": 0.6,
              "x0": 'Total Bill'
          }],
          "layout" : {
              "title": "",
              "yaxis": {
                  "zeroline": False,
              }
          }
      }

      viz = logger.viz
      viz.plotlyplot(fig)
      # py.iplot(fig, filename = 'violin', validate = False)
      qweqwe

    if(p.requires_grad) and ("bias" not in n):
      # if p.grad is None:
      #   n_none += 1
      #   # layers.append(n)
      # else:
      #   n_grad += 1

      layers.append(n)
      # print(n)
      # print(p.grad.shape)
      # print(p.shape)
      # qweq
      ave_grads.append(p.grad.abs().mean())


  # print(n_none, n_grad)
  # print(layers)
  qweqw

  plt.plot(ave_grads, alpha=0.3, color="b")
  plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
  plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
  plt.xlim(xmin=0, xmax=len(ave_grads))
  plt.xlabel("Layers")
  plt.ylabel("average gradient")
  plt.title("Gradient flow")
  plt.grid(True)
  plt.show()


  def pause():
    input("PRESS KEY TO CONTINUE.")

  pause()

if __name__ == '__main__':
  train()