from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import torchvision.transforms as transforms
import json, random, torch
import imgaug as ia
import imgaug.augmenters as iaa
from transform import *

# TODO: benchmark?
# http://icvl.ee.ic.ac.uk/DescrWorkshop/index.html#Challenge

def setup_traindata(config, transformer):
  def collate(batch):
    im_sizes = config.image_input_size
    im_size = random.randint(im_sizes[0], im_sizes[1])
    uniform_size = transforms.Compose([
      transforms.Resize((im_size, im_size)),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    original_ims = [uniform_size(b[0]) for b in batch]
    transformed_ims = [uniform_size(b[1]) for b in batch]

    return torch.stack(original_ims), torch.stack(transformed_ims)

  batch_size = config.batch_size
  dataset = TripletDataset(config.dataset, transformer)
  return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=config.num_workers, collate_fn=collate, drop_last=True)

def setup_valdata(config, transformer=None):
  normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])

  def collate(batch):
    item = list(batch[0])
    item[0] = normalize(item[0]).unsqueeze(0)
    return item

  index_file = f'{config.validation_dataset}/index.json'
  # dataset = CopyDataset(index_file, config)
  dataset = OnlineTransformDataset(config.validation_dataset, transformer)
  return DataLoader(dataset, batch_size=1, collate_fn=collate, num_workers=config.num_workers)

def setup_duplicatedata(config):
  normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    ])

  def collate(batch):
    item = list(batch[0])
    item[0] = normalize(item[0]).unsqueeze(0)
    return item

  dataset = ImageDataset(config.validation_dataset)
  dataloader =  DataLoader(dataset, batch_size=1, collate_fn=collate, num_workers=config.num_workers)
  return dataset, dataloader


class TripletDataset(Dataset):
  def __init__(self, im_dirs, transform, n_fakes=1):
    self.n_fakes = n_fakes
    self.transform = transform
    self.to_tensor = transforms.ToTensor()
    self.image_files = []
    
    im_types = ['.jpg', '.png']
    is_image = lambda path: path.suffix in im_types

    if type(im_dirs) == str:
      im_dirs = [im_dirs]

    for im_dir in im_dirs:
      assert Path(im_dir).exists(), "Directory doesn't exist"
      image_files = [f for f in Path(im_dir).glob('**/*') if is_image(f)]
      self.image_files.extend(image_files)

    assert self.image_files,'{} dataset is empty'.format(im_dir)

  def __len__(self):
    return len(self.image_files)

  def __getitem__(self, index):
    im_path = self.image_files[index]

    im = Image.open(im_path)
    if im.mode != 'RGB': # Handle black & white images
      im = im.convert(mode='RGB') 

    # uniform_size = transforms.Resize((self.im_size, self.im_size))
    # im = uniform_size(im)

    transformed_ims = []
    for _ in range(self.n_fakes):
      t_im = self.transform(im)
      # t_im = uniform_size(t_im)
      # transformed_ims.append(self.to_tensor(t_im))
      transformed_ims.append(t_im)

    # Returns original image, list of transformed images
    # return self.to_tensor(im), transformed_ims
    return im, transformed_ims[0]

class CopyDataset(Dataset):
  def __init__(self, index_file, config):
    self.data_root = str(Path(index_file).parent)
    with open(index_file) as infile:
      self.index_json = json.load(infile)

    # TODO: Asssert that there is both query and db
    assert self.index_json,'{} dataset is empty'.format(index_file)

  def __len__(self):
    return len(self.index_json)

  def __getitem__(self, index):
    data = self.index_json[index]

    open_image = lambda p: Image.open('{}/{}'.format(self.data_root, p))
    im = open_image(data['path'])
    im = im.convert('RGB')

    # TODO: Want to remove uniform size later
    uniform_size = transforms.Resize((300, 300))
    return uniform_size(im), data['im_type'], data['match_id'], data['im_id']
    # return im, data['im_type'], data['match_id'], data['im_id']

class ImageDataset(Dataset):
  def __init__(self, im_dirs):
    self.image_files = []
    im_types = ['.jpg', '.png']
    is_image = lambda path: path.suffix in im_types

    if type(im_dirs) == str:
      im_dirs = [im_dirs]

    for im_dir in im_dirs:
      assert Path(im_dir).exists(), "Directory doesn't exist"
      image_files = [f for f in Path(im_dir).glob('**/*') if is_image(f)]
      self.image_files.extend(image_files)

    assert self.image_files,'{} dataset is empty'.format(im_dir)

  def __len__(self):
    return len(self.image_files)

  def __getitem__(self, index):
    path = self.image_files[index]
    im = Image.open(path)
    im = im.convert('RGB')

    uniform_size = transforms.Resize((300, 300))
    return uniform_size(im), None, None, index

class OnlineTransformDataset(Dataset):
  def __init__(self, im_folder, transformer):
    self.ims = list(Path(im_folder).iterdir())
    self.transformer = transformer

  def __len__(self):
    return len(self.ims) * 2

  def __getitem__(self, index):
    even_index = index % 2 == 0

    # Match id
    match_id = int(index / 2)

    # Image
    im_path = self.ims[match_id]
    im = Image.open(im_path)
    im = im.convert('RGB')
    if even_index:
      im = self.transformer(im)

    # Query/database type
    if even_index:
      im_type = 'query'
    else:
      im_type = 'database'

    return im, im_type, match_id, index

if __name__ == '__main__':
  import visdom, torch, random
  from transform import *
  from config.config_util import choose_config
  import imgaug as ia

  seed = random.randint(0, 10000)
  ia.seed(seed)

  def clear_envs(viz):
    [viz.close(env=env) for env in viz.get_env_list()] # Kills wind

  viz = visdom.Visdom(port='6006')
  clear_envs(viz)

  # transformer = AllTransformer()
  transformer = CropTransformer()
  # transformer = RotateTransformer()
  # transformer = JpgTransformer()
  # config = choose_config('laptop')
  config = choose_config('colab')
  
  # data_path = 'datasets/copydays/original'
  data_path = 'datasets/places365/validation'
  dataset = TripletDataset(data_path, transformer)

  data_inds = list(range(len(dataset)))
  random.shuffle(data_inds)
  n_to_show = 5
  data_inds = data_inds[:n_to_show]
  to_tensor = transforms.ToTensor()

  def transform_im(ind):
    im, t_im = dataset[ind]
    return to_tensor(im), to_tensor(t_im)

  for i in data_inds:
    im, t_im = transform_im(i)
    # viz.images([im, t_im])
    viz.image(im)
    viz.image(t_im)
