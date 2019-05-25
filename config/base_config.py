import pyjokes, random
from datetime import datetime as dtime
from collections import OrderedDict

class DefaultConfig():
  def __init__(self, config_str):
    # ~~~~~~~~~~~~~~ General Parameters ~~~~~~~~~~~~~~
    # An optional comment to differentiate this run from others
    self.save_comment = pyjokes.get_joke()
    print('\n{}\n'.format(self.save_comment))

    # Start time to keep track of when the experiment was run
    self.start_time = dtime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    # Use GPU. Set to False to only use CPU
    self.use_gpu = True

    self.max_val_batches = 10

    self.num_workers = 0

    # The config name
    self.config = config_str 

    self.dataset = 'datasets/cifar_bigger'

    self.n_model_features = 10

  def get_parameters(self):
    return OrderedDict(sorted(vars(self).items()))

  def __str__(self): # TODO return str
    # class name, etc
    return str(vars(self))

class Laptop(DefaultConfig):
  def __init__(self, config_str):
    super().__init__(config_str)
    ''' Change default parameters here. Like this
    self.seed = 666          ____
      ________________________/ O  \___/  <--- Python <3
     <_#_#_#_#_#_#_#_#_#_#_#_#_____/   \
    '''
    self.use_gpu = False

class Colab(DefaultConfig):
  def __init__(self, config_str):
    super().__init__(config_str)
    self.max_val_batches = 100
    self.num_workers = 16
    self.dataset = 'datasets/imagenet'
    self.n_model_features = 512

