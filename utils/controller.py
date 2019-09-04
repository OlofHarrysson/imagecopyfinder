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
from utils.loss import PositiveCosineLoss, ZeroCosineLoss

# TODO: Can have a prepare data colab file that prepares the data and puts it in gdrive. It could download a dataset online. Then a user can upload its dataset to his/hers gdrive. Would be able to run the project from any computer without installation

def clear_output_dir():
  [p.unlink() for p in Path('output').iterdir()]

def init_training(model, config):
  torch.backends.cudnn.benchmark = True # Optimizes cudnn
  model.to(model.device) # model -> CPU/GPU

  # Optimizer & Scheduler
  # TODO CyclicLR
  # params = filter(lambda p: p.requires_grad, model.parameters())

  my_list = ['feature_extractor.sim_weights']
  weight_params = [kv[1] for kv in model.named_parameters() if kv[0] in my_list]
  base_params = [kv[1] for kv in model.named_parameters() if kv[0] not in my_list]

  optimizer = torch.optim.SGD([
    {'params': weight_params, 'lr': 5e-1},
    {'params': base_params},

    ], lr=config.start_lr, momentum=0.9)
  # optimizer = torch.optim.SGD(params, lr=config.start_lr, momentum=0.9)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.optim_steps/config.lr_step_frequency, eta_min=config.end_lr)

  return optimizer, scheduler

def train(model, config):
  # clear_output_dir()
  optimizer, lr_scheduler = init_training(model, config)
  logger = Logger(config)

  # TODO: Check which images it thinks are similar from e.g. copydays.

  # transformer = AllTransformer()
  # transformer = JpgTransformer()
  # transformer = RotateTransformer()
  # transformer = FlipTransformer()
  # transformer = RotateCropTransformer()
  transformer = CropTransformer()
  validator = Validator(model, logger, config, transformer)

  margin = 5
  triplet_loss_fn = TripletMarginLoss(margin, p=config.distance_norm, swap=True)
  neg_cos_loss_fn = ZeroCosineLoss(margin=0.1)
  pos_cos_loss_fn = PositiveCosineLoss(margin=0.1)

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
      # triplet_loss = triplet_loss_fn(anchors, positives, negatives)
      anchors, positives = scale_embeddings(anchors, positives, model)
      anchors, negatives = scale_embeddings(anchors, negatives, model)

      # Cosine similarity loss
      cos_match_loss = pos_cos_loss_fn(anchors, positives)
      cos_not_match_loss = neg_cos_loss_fn(anchors, negatives)

      # loss_dict = dict(triplet=triplet_loss, cos_pos=cos_match_loss, cos_neg=cos_not_match_loss)
      loss_dict = dict(cos_pos=cos_match_loss, cos_neg=cos_not_match_loss)

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
      logger.log_weights(model.feature_extractor.sim_weights)

      
      # Frees up GPU memory
      del data; del outputs


def scale_embeddings(embs1, embs2, model):
  ''' Normally, similarity could be dominated by one or a few activations. If these activations aren't in both images, then similarity will be low. We want to emphasise on the activations that DO match (similar is worth more than dissimilar) '''

  # TODO: Weights never grow very large. That means that even if the diff is very low for an embedding val, that proximity doesn't help very much in finding duplicates. The weight selection is based on the diff, but in fact we should really consider the magnitude of the embeddings before we select weights. The really small embeddings will be closer to one another than two mroe %-similar big embeddings. Can we find a way to also incorporate this info? In a data driven matter?

  diff = torch.abs(embs1 - embs2)
  sort_vals, sort_inds = torch.sort(diff)
  
  sort_inds = sort_inds.view((1, -1))
  weights = model.feature_extractor.sim_weights.view((1, -1))

  s_weights = weights[:, sort_inds]
  s_weights = s_weights.view(embs1.shape[0], -1)

  s_q = s_weights * embs1
  s_db = s_weights * embs2

  return s_q, s_db

if __name__ == '__main__':
  train()