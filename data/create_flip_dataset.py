from pathlib import Path
from PIL import Image
import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
import random


def resize_ims(im, cropped_im):
  im.thumbnail((600, 600))
  cropped_im.thumbnail((600, 600))
  return im, cropped_im

def create_transformed(im_path):
  im = Image.open(im_path)

  aug = iaa.SomeOf((1, 2), [
    iaa.Fliplr(1.0),
    iaa.Flipud(1.0),
    ])

  transf_im = aug.augment_image(np.array(im))
  transf_im = Image.fromarray(transf_im)

  return im, transf_im

def main():
  im_dir = Path('../datasets/places365/validation_val/original_src')
  out_dir = Path('../datasets/places365/validation_val/flip')

  (out_dir / 'original').mkdir(exist_ok=True, parents=True) 
  (out_dir / 'fake').mkdir(exist_ok=True) 

  for ind, src_im in enumerate(im_dir.iterdir()):
    if src_im.name[0] == '.': # Is hidden file
      continue

    im, transf_im = create_transformed(src_im)
    im, transf_im = resize_ims(im, transf_im)

    im.save(out_dir / 'original' / src_im.name)
    transf_im.save(out_dir / 'fake' / src_im.name)


if __name__ == '__main__':
  seed = random.randint(0, 10000)
  ia.seed(seed)
  main()