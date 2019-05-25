import argparse
from utils.controller import train
from config.config_util import choose_config
from resnet import Resnet18

def parse_args():
  p = argparse.ArgumentParser()

  configs = ['laptop', 'colab']
  p.add_argument('--config', type=str, default='laptop', choices=configs, help='What config file to choose')

  args, unknown = p.parse_known_args()
  return args.config

def main(config_str):
  config = choose_config(config_str)

  # Create model
  model = Resnet18(config)
  train(model, config)

if __name__ == '__main__':
  config_str = parse_args()
  main(config_str)