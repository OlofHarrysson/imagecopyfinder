import torch
from utils.utils import ProgressbarWrapper as Progressbar
from data import TripletDataset
from torch.utils.data import DataLoader
from triplet import create_triplets, create_doublets
from logger import Logger
from utils.validator import Validator
from transform import Transformer
from pathlib import Path
from resnet import DistanceNet

# TODO: I can increase the augmentation depending on how many easy/hard there are. Possibly also increase margin?
# TODO: Change distance norm 2->1? I want the system to match because there are a lot of features that are close, not fuck up because one feature is bad and dominates the others. Could even select the top% features that matchest best.
# TODO: Could this be used as a pretraining system for triplet loss

# TODO: Create some more elaborate transforms. Combine them.
# TODO: Create a validation dataset. contains both real and adverserial examples, labeled so we can measure accuracy

# TODO: Config. Default, laptop & colab
# TODO: Can have a prepare data colab file that prepares the data and puts it in gdrive. It could download a dataset online. Then a user can upload its dataset to his/hers gdrive. Would be able to run the project from any computer without installation


def clear_output_dir():
  [p.unlink() for p in Path('output').iterdir()]


def init_training(model):
  torch.backends.cudnn.benchmark = True # Optimizes cudnn
  model.to(model.device) # model -> CPU/GPU

  # Optimizer & Scheduler
  # add_params = lambda m1, m2: list(m1.parameters()) + list(m2.parameters())
  # all_params = add_params(feature_extractor, distance_net)
  optimizer = torch.optim.Adam(model.parameters(), weight_decay=5e-4, lr=1e-3)
  return optimizer


def train(model, config):
  # clear_output_dir()
  optimizer = init_training(model)
  logger = Logger()
  validator = Validator(model, logger, config)
  transformer = Transformer()
  margin = 1.0
  # loss_f = torch.nn.TripletMarginLoss(margin)
  loss_fn = torch.nn.BCELoss()

  # Data
  batch_size = 4
  dataset = TripletDataset(config.dataset, transformer, n_fakes=1)
  dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=config.num_workers)

  # Init progressbar
  n_batches = len(dataloader)
  n_epochs = 100
  pbar = Progressbar(n_epochs, n_batches)

  optim_steps = 0
  val_freq = 100
  # Training loop starts here
  for epoch in pbar(range(1, n_epochs + 1)):
    for batch_i, data in enumerate(dataloader, 1):
      pbar.update(epoch, batch_i)

      # Validation
      # if optim_steps % val_freq == val_freq - 1:
      # if optim_steps % val_freq == 0:
        # validator.validate(optim_steps)

      optimizer.zero_grad()
      original, transformed = data
      transformed = transformed[0]

      # original = torch.randn((4, 3, 512, 512))
      # transformed = torch.randn((4, 3, 512, 512))

      inputs = torch.cat((original, transformed))
      outputs = model(inputs)

      
      # outputs = feature_extractor(inputs)
      # original_out, transf_out = outputs.chunk(2)

      # anchors, positives, negatives = create_triplets(original_out, transf_out)
      # a_2_p = torch.cat((anchors, positives), dim=1) # Dist -> 0
      # a_2_n = torch.cat((anchors, negatives), dim=1) # Dist -> 1

      # distance_input = torch.cat((a_2_p, a_2_n))
      # distance_output = distance_net(distance_input)

      # pos_distance, neg_distance = distance_output.chunk(2)
      pos_distance, neg_distance = outputs.chunk(2)
      pos_loss = loss_fn(pos_distance, torch.zeros_like(pos_distance))
      neg_loss = loss_fn(neg_distance, torch.ones_like(neg_distance))
      loss = pos_loss + neg_loss
      # loss = loss_f(anchors, positives, negatives)
      print(loss)

      loss.backward()
      optimizer.step()
      optim_steps += 1

      # logger.easy_or_hard(anchors, positives, negatives, margin, optim_steps)
      logger.log_distance_accuracy(pos_distance, neg_distance, optim_steps)
      # Frees up GPU memory
      del data; del outputs


if __name__ == '__main__':
  train()