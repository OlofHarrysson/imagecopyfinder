from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import torchvision.transforms as transforms
import json

# TODO: benchmark?
# https://archive.org/details/ukbench
# http://icvl.ee.ic.ac.uk/DescrWorkshop/index.html#Challenge

class TripletDataset(Dataset):
  def __init__(self, im_dirs, transform, n_fakes):
    self.n_fakes = n_fakes
    self.transform = transform
    self.to_tensor = transforms.ToTensor()
    self.image_files = []

    is_hidden_file = lambda path: path.name[0] == '.'
    if type(im_dirs) == str:
      im_dirs = [im_dirs]

    for im_dir in im_dirs:
      print(im_dir)
      assert Path(im_dir).exists(), "Directory doesn't exist"
      all_files = Path(im_dir).iterdir()
      image_files = [p for p in all_files if not is_hidden_file(p)]
      self.image_files.extend(image_files)

    assert self.image_files,'{} dataset is empty'.format(im_dir)

  def __len__(self):
    return len(self.image_files)

  def __getitem__(self, index):
    im_path = self.image_files[index]

    im = Image.open(im_path)
    if im.mode != 'RGB': # Handle black & white images
      im = im.convert(mode='RGB') 

    uniform_size = transforms.Resize((300, 300))
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