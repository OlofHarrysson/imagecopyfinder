import argparse
from utils.controller import train
from config.config_util import choose_config
from models.distancenet import DistanceNet
from utils.utils import seed_program

def parse_args():
  p = argparse.ArgumentParser()

  configs = ['laptop', 'colab']
  p.add_argument('--config', type=str, default='laptop', choices=configs, help='What config file to choose')

  args, unknown = p.parse_known_args()
  return args.config

def main(config_str):
  config = choose_config(config_str)
  seed_program(config.seed)
  model = DistanceNet(config)
  if config.model_path:
    model.load(config.model_path)
  
  train(model, config)

if __name__ == '__main__':
  config_str = parse_args()
  main(config_str)