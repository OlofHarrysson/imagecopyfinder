import torch, math, random
from utils.utils import ProgressbarWrapper as Progressbar
from data import setup_traindata
from triplet import create_triplets, create_doublets
from logger import Logger
from utils.validator import Validator
from transform import *
from pathlib import Path
import torchvision.transforms as transforms
import numpy as np
from torch.nn import TripletMarginLoss

# TODO: Can have a prepare data colab file that prepares the data and puts it in gdrive. It could download a dataset online. Then a user can upload its dataset to his/hers gdrive. Would be able to run the project from any computer without installation


# TODO: After all conv2d layers, do adaptive pooling and continue with conv1d layers. At this point I don't care about spatial info anymore. Perhaps add a fc in after the adaptive pooling. But the reason is that conv layers use less parameters. Can try with fc layers as well.

def clear_output_dir():
  [p.unlink() for p in Path('output').iterdir()]


def init_training(model, config):
  torch.backends.cudnn.benchmark = True # Optimizes cudnn
  model.to(model.device) # model -> CPU/GPU

  # Optimizer & Scheduler
  # TODO CyclicLR
  params = filter(lambda p: p.requires_grad, model.parameters())
  # optimizer = torch.optim.Adam(params, lr=config.start_lr)
  optimizer = torch.optim.SGD(params, lr=config.start_lr, momentum=0.9)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.optim_steps/config.lr_step_frequency, eta_min=config.end_lr)

  return optimizer, scheduler


def train(model, config):
  # clear_output_dir()
  optimizer, lr_scheduler = init_training(model, config)
  logger = Logger(config)

  transformer = AllTransformer()
  # transformer = CropTransformer()
  # transformer = RotateTransformer()
  # transformer = FlipTransformer()
  validator = Validator(model, logger, config, transformer)

  margin = 5
  triplet_loss_fn = TripletMarginLoss(margin, p=config.distance_norm, swap=True)
  cos_loss_fn = torch.nn.CosineEmbeddingLoss(margin=0.1) # margin helps with separating the pos/neg in violin loss for fliptransformer. The negative tail is shorter.
  # make own cosine loss which has positive margin. and another which has margin in both directions around 0

  # Data
  dataloader = setup_traindata(config, transformer)

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
      if optim_steps % val_freq == 0:
        validator.validate(optim_steps)

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

      # Cosine similarity loss
      y_size = anchors.size(0)
      y = torch.ones(y_size).to(model.device)
      cos_match_loss = cos_loss_fn(anchors, positives, y)
      cos_not_match_loss = cos_loss_fn(anchors, negatives, -y)
      cos_not_match_loss *= 60

      loss_dict = dict(triplet=triplet_loss, cos_pos=cos_match_loss, cos_neg=cos_not_match_loss)

      loss = sum(loss_dict.values())
      loss.backward()
      optimizer.step()
      optim_steps += 1

      corrects = model.corrects(transf_emb, original_emb)
      logger.easy_or_hard(anchors, positives, negatives, margin, optim_steps)
      logger.log_loss(loss, optim_steps)
      logger.log_loss_percent(loss_dict, optim_steps)
      logger.log_corrects(corrects, optim_steps)
      logger.log_cosine(anchors, positives, negatives)
      # logger.log_p(model.feature_extractor.pool.p, optim_steps)
      
      # Frees up GPU memory
      del data; del outputs

if __name__ == '__main__':
  train()