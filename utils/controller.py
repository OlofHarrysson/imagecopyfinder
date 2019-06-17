import torch, math
from utils.utils import ProgressbarWrapper as Progressbar
from data import TripletDataset
from torch.utils.data import DataLoader
from triplet import create_triplets, create_doublets
from logger import Logger
from utils.validator import Validator
from transform import Transformer, CropTransformer
from pathlib import Path
from resnet import DistanceNet

# TODO: I can increase the augmentation depending on how many easy/hard there are. Possibly also increase margin?
# TODO: Change distance norm 2->1? I want the system to match because there are a lot of features that are close, not fuck up because one feature is bad and dominates the others. Could even select the top% features that matchest best.
# TODO: Could this be used as a pretraining system for triplet loss


# TODO: Can have a prepare data colab file that prepares the data and puts it in gdrive. It could download a dataset online. Then a user can upload its dataset to his/hers gdrive. Would be able to run the project from any computer without installation


def clear_output_dir():
  [p.unlink() for p in Path('output').iterdir()]


def init_training(model, config):
  torch.backends.cudnn.benchmark = True # Optimizes cudnn
  model.to(model.device) # model -> CPU/GPU

  # Optimizer & Scheduler
  # TODO CyclicLR
  # optimizer = torch.optim.Adam(model.parameters(), weight_decay=5e-4, lr=config.start_lr)
  optimizer = torch.optim.Adam(model.similarity_net.parameters(), lr=config.start_lr)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.optim_steps/config.lr_step_frequency, eta_min=config.end_lr)

  return optimizer, scheduler


def train(model, config):
  # clear_output_dir()
  optimizer, lr_scheduler = init_training(model, config)
  logger = Logger(config)
  validator = Validator(model, logger, config)
  # transformer = Transformer()
  transformer = CropTransformer()
  margin = 5
  triplet_loss_fn = torch.nn.TripletMarginLoss(margin, p=config.distance_norm)
  similarity_loss_fn = torch.nn.BCELoss()
  # similarity_loss_fn = torch.nn.MSELoss()

  # Data
  batch_size = config.batch_size
  dataset = TripletDataset(config.dataset, transformer, config)
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=config.num_workers)

  # Init progressbar
  n_batches = len(dataloader)
  n_epochs = math.ceil(config.optim_steps / n_batches)
  pbar = Progressbar(n_epochs, n_batches)

  def get_lr(optimiz):
    for param_group in optimiz.param_groups:
      return param_group['lr']

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
      transformed = transformed[0]

      inputs = torch.cat((original, transformed))
      outputs = model(inputs)
      original_emb, transf_emb = outputs
      anchors, positives, negatives = create_triplets(original_emb, transf_emb)
      # print(f'Anchors: {anchors}, Pos: {positives}, Neg: {negatives}')
      
      a_p, a_n = model.cc_similarity_net(anchors, positives, negatives)
      # print(f'Positives: {a_p.mean()}, Negatives: {a_n.mean()}')
      # print(f'Positives: {a_p}, Negatives: {a_n}')
# 
      # triplet_loss = triplet_loss_fn(anchors, positives, negatives)
      positives = similarity_loss_fn(a_p, torch.ones_like(a_p))
      negatives = similarity_loss_fn(a_n, torch.zeros_like(a_n))

      # print(f'Positive loss: {positives}, Negative: {negatives}')
      
      # positives = similarity_loss_fn(a_p, torch.ones_like(a_p))
      # negatives = similarity_loss_fn(a_n, torch.ones_like(a_n))
      loss = positives + negatives
      # loss = negatives
      # loss = triplet_loss

      corrects = model.corrects(transf_emb, original_emb)
      loss.backward()
      optimizer.step()
      optim_steps += 1

      # print(original_emb, anchors)
      # qweqwe
      # TODO: it seems like i cant overfit. a_p, a_n are approaching good values but corrects doesn't. Why?
      # asdasd

      # if optim_steps % 50 == 0:
      # logger.easy_or_hard(anchors, positives, negatives, margin, optim_steps)
      logger.log_loss(loss, optim_steps)
      logger.log_corrects(corrects, optim_steps)
      # logger.log_lr(get_lr(optimizer), optim_steps)
      
      # Frees up GPU memory
      del data; del outputs


if __name__ == '__main__':
  train()