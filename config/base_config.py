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

    # self.validation_dataset = 'datasets/copydays_crop'
    # self.validation_dataset = 'datasets/places365_crop'
    # self.validation_dataset = 'datasets/places365_flip'
    self.validation_dataset = 'datasets/places365/validation_val/original_src'
    # self.validation_dataset = 'datasets/copydays/original'

    self.n_model_features = 10

    # Range of Data input size image sizes
    # self.image_input_size = (200, 500)
    self.image_input_size = (300, 300)

    # self.validation_im_size = 

    self.batch_size = 3

    self.pretrained = True

    self.distance_norm = 2

    self.start_lr = 5e-2
    self.end_lr = 1e-3

    self.optim_steps = 20000
    self.lr_step_frequency = 100

    self.validation_freq = 200

    self.top_x = 3

    # Seed to create reproducable training results
    self.seed = random.randint(0, 2**32 - 1)


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
    self.image_input_size = (30, 35)
    self.n_model_features = 3
    # self.n_model_features = 400
    self.max_val_batches = 10
    # self.max_val_batches = 30
    # self.batch_size = 3
    self.batch_size = 16

    self.sample = True

    self.dataset = 'datasets/cifar_50'
    # self.dataset = 'datasets/cifar_sample'

    self.seed = 9

    # self.n_model_features = 8192


class Colab(DefaultConfig):
  def __init__(self, config_str):
    super().__init__(config_str)
    self.num_workers = 16
    self.dataset = 'datasets/places365_big' # Without the places365 crop used for validation

    # self.n_model_features = 512
    # self.n_model_features = 2048
    self.n_model_features = 8192

    self.top_x = int(self.n_model_features * 0.3)

    self.batch_size = 32

    # self.pretrained = False

    self.validation_freq = 50




