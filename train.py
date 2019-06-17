import argparse
from utils.controller import train
from config.config_util import choose_config
from resnet import Resnet18, DistanceNet
from utils.utils import seed_program

def parse_args():
  p = argparse.ArgumentParser()

  configs = ['laptop', 'colab']
  p.add_argument('--config', type=str, default='laptop', choices=configs, help='What config file to choose')

  args, unknown = p.parse_known_args()
  return args.config

def main(config_str):
  config = choose_config(config_str)
  seed_program()

  # Create model
  # feature_extractor = Resnet18(config)
  # distance_net = DistanceNet(config)
  model = DistanceNet(config)

  train(model, config)

if __name__ == '__main__':
  config_str = parse_args()
  main(config_str)