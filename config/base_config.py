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

    self.max_val_batches = 999

    self.num_workers = 0

    # The config name
    self.config = config_str 

    self.dataset = 'datasets/cifar_bigger'

    self.validation_dataset = 'datasets/copydays_crop'

    self.n_model_features = 10

    # Data input size
    self.image_input_size = 32

    self.batch_size = 3

    self.pretrained = True

    self.distance_norm = 1

    self.start_lr = 1e-3
    self.end_lr = 1e-4

    self.optim_steps = 20000
    self.lr_step_frequency = 100

    self.validation_freq = 200

    self.top_x = 3



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
    self.n_model_features = 4
    self.max_val_batches = 10
    # self.max_val_batches = 30
    self.batch_size = 3


class Colab(DefaultConfig):
  def __init__(self, config_str):
    super().__init__(config_str)
    self.max_val_batches = 100
    self.num_workers = 16
    # self.dataset = 'datasets/imagenet'
    self.dataset = 'datasets/places365/validation'

    self.n_model_features = 512
    self.top_x = int(self.n_model_features / 4)

    self.image_input_size = 300
    self.batch_size = 16

    self.pretrained = False

    self.validation_freq = 60



