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

def create_crops(im_path):
  im = Image.open(im_path)

  minc, maxc = 0.05, 0.3
  crop_percent = ([minc, maxc], [minc, maxc], [minc, maxc], [minc, maxc])
  aug = iaa.Crop(percent=crop_percent, keep_size=False)
  cropped_im = aug.augment_image(np.array(im))
  cropped_im = Image.fromarray(cropped_im)

  return im, cropped_im

def main():
  # im_dir = Path('../datasets/copydays/original_src')
  # out_dir = Path('../datasets/copydays_crop')

  im_dir = Path('../datasets/places365/validation_val/original_src')
  out_dir = Path('../datasets/places365/validation_val/cropped')

  (out_dir / 'original').mkdir(exist_ok=True, parents=True) 
  (out_dir / 'fake').mkdir(exist_ok=True) 

  for ind, src_im in enumerate(im_dir.iterdir()):
    if src_im.name[0] == '.': # Is hidden file
      continue

    im, cropped_im = create_crops(src_im)
    im, cropped_im = resize_ims(im, cropped_im)

    im.save(out_dir / 'original' / src_im.name)
    cropped_im.save(out_dir / 'fake' / src_im.name)


if __name__ == '__main__':
  seed = random.randint(0, 10000)
  ia.seed(seed)
  main()