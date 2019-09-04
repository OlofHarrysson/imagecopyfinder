import argparse, torch
from utils.controller import train
from config.config_util import choose_config
from models.distancenet import DistanceNet
from utils.utils import seed_program
from data import setup_duplicatedata
from models.dataclasses import Entry
from collections import OrderedDict, defaultdict
from torchvision.transforms.functional import to_pil_image, to_tensor
import torch.nn.functional as F
from logger import Logger
from PIL import Image, ImageDraw
import numpy as np
import torch.nn as nn




def parse_args():
  p = argparse.ArgumentParser()

  configs = ['duplicate']
  p.add_argument('--config', type=str, default='duplicate', choices=configs, help='What config file to choose')

  args, unknown = p.parse_known_args()
  return args.config

def main(config_str):
  config = choose_config(config_str)
  seed_program(config.seed)
  model = DistanceNet(config)
  if config.model_path:
    model.load(config.model_path)
  model.eval()

  dataset, dataloader = setup_duplicatedata(config)
  logger = Logger(config)

  # TODO: Temp
  # for ind, data in enumerate(dataloader, 1):
  #   im, im_type, match_id, im_id = data

  #   if ind != 25:
  #     continue

  #   real_im = dataset[ind][0]
  #   logger.log_image(to_tensor(real_im), im_id)
  #   # pool = nn.AdaptiveAvgPool2d((1, 1))
  #   pool = nn.AdaptiveMaxPool2d((1, 1))


  #   output = model.predict_embedding(im).cpu()
  #   avg = pool(output)
  #   logger.log_violin(avg.squeeze(), 'average')
  #   up = nn.Upsample(100, mode='bilinear')
  #   # output = up(output)

  #   for filter_i, oo in enumerate(output.squeeze()):
  #     oo = oo / oo.sum(0).expand_as(oo)

  #     # logger.log_image(oo, filter_i)

  #     if filter_i == 100:
  #       asdasdasd

  # qwe



  embeddings = calc_embeddings(dataloader, model, config)
  matches_dict = defaultdict(list)
  entry_keys = list(embeddings.keys())

  for entry, emb in embeddings.items():
    # Query & database entry similarities
    similarity_dict = model.similarities(emb, embeddings)
    for metric_name, similarities in similarity_dict.items():

      # Finds best match & rank of the prediction
      # _, similarities_sorted = similarities.topk(similarities.size(0))
      # for rank_number, sim_ind in enumerate(similarities_sorted):
      sim_vals, sim_inds = similarities.topk(similarities.size(0))
      for rank_number, simi in enumerate(zip(sim_vals, sim_inds)):
        sim_value, sim_ind = simi
        if rank_number == 0: # Always best matches with itself
          continue

        top = 5
        if rank_number == top + 1: # Top results
          break

        # matches_dict[entry.im_id].append(sim_ind.item())
        matches_dict[entry.im_id].append((sim_ind.item(), sim_value.item()))
  
  for query_id, matches in matches_dict.items():
    image_ids = [(query_id, 0)] + matches
    concat_images(image_ids, dataset, logger)

def concat_images(im_ids, dataset, logger, im_size=200):
  concat_im = torch.tensor([])
  for im_id in im_ids:
    im_id, match_val = im_id
    im, *_ = dataset[im_id]
    im.thumbnail((im_size, im_size))

    if match_val: # Disregards the first image
      d = ImageDraw.Draw(im)
      d.text((5,5), f'Match: {match_val:.2f}%')

    im = squarify(im)
    concat_im = torch.cat((concat_im, im), dim=2)

  logger.log_image(concat_im, im_ids[0])

def squarify(im):
  im = to_tensor(im)
  c, h, w = im.size()
  if h > w:
    padding = (h-w, 0, 0, 0)
  else:
    padding = (0, 0, w-h, 0)
  return F.pad(im, padding)


def calc_embeddings(dataloader, model, config):
  embeddings = OrderedDict()
  for ind, data in enumerate(dataloader, 1):
    if ind > config.max_val_batches:
      break

    im, im_type, match_id, im_id = data
    entry = Entry(im_type, match_id, im_id)

    with torch.no_grad():
      embeddings[entry] = model.predict_embedding(im).cpu() # TODO: GPU?

  return embeddings

if __name__ == '__main__':
  config_str = parse_args()
  main(config_str)