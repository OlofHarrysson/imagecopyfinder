from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import torchvision.transforms as transforms
import json

# TODO: benchmark?
# https://archive.org/details/ukbench
# http://icvl.ee.ic.ac.uk/DescrWorkshop/index.html#Challenge


#TODO Other training datasets
# SAVOIAS 1400 ims https://github.com/esaraee/Savoias-Dataset
# Tencent 17M ims, 78k in val. From imagenet+openimages https://github.com/Tencent/tencent-ml-images#download-images
# Places. Looked diverse and easy to download. Big dataset :)


# TODO: Other val datasets
# Oxford5K, Paris6K of buildings
# This one? https://github.com/chenjun082/holidays


class TripletDataset(Dataset):
  def __init__(self, im_dirs, transform, config, n_fakes=1):
    self.n_fakes = n_fakes
    self.transform = transform
    self.to_tensor = transforms.ToTensor()
    self.image_files = []
    self.im_size = config.image_input_size
    
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

    uniform_size = transforms.Resize((self.im_size, self.im_size))
    im = uniform_size(im)

    transformed_ims = []
    for _ in range(self.n_fakes):
      t_im = self.transform(im)
      t_im = uniform_size(t_im)
      transformed_ims.append(self.to_tensor(t_im))

    # Returns original image, list of transformed images
    return self.to_tensor(im), transformed_ims

class CopyDataset(Dataset):
  def __init__(self, index_file):
    self.data_root = str(Path(index_file).parent)
    with open(index_file) as infile:
      self.index_json = json.load(infile)

    # TODO: Asssert tht there is both query and db
    assert self.index_json,'{} dataset is empty'.format(index_file)

  def __len__(self):
    return len(self.index_json)

  def __getitem__(self, index):
    data = self.index_json[index]

    open_image = lambda p: Image.open('{}/{}'.format(self.data_root, p))
    im = open_image(data['path'])
    
    return im, data['im_type'], data['match_id'], data['im_id']



if __name__ == '__main__':
  import visdom, torch, random
  from transform import Transformer, CropTransformer
  from config.config_util import choose_config
  import imgaug as ia

  seed = random.randint(0, 10000)
  ia.seed(seed)

  def clear_envs(viz):
    [viz.close(env=env) for env in viz.get_env_list()] # Kills wind

  viz = visdom.Visdom(port='6006')
  clear_envs(viz)

  # transformer = Transformer()
  transformer = CropTransformer()
  # config = choose_config('laptop')
  config = choose_config('colab')
  dataset = TripletDataset('datasets/places365/validation', transformer, config)

  data_inds = list(range(len(dataset)))
  random.shuffle(data_inds)
  n_to_show = 5
  data_inds = data_inds[:n_to_show]

  def transform_im(ind):
    im, t_im = dataset[ind]
    t_im = t_im[0]
    return torch.cat((im, t_im), dim=1)

  for i in data_inds:
    im = transform_im(i)
    viz.image(im)