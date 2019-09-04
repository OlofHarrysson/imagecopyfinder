import torchvision.transforms as transforms
from PIL import Image
import imgaug as ia
import imgaug.augmenters as iaa
# from imgaug import parameters as iap
import numpy as np


class Transformer():
  def numpy_transform(self, im):
    return self.seq.augment_image(np.array(im))

  def grid(self, im):
    self.seq.show_grid(im, cols=6, rows=4)

  def __call__(self, im):
    augmented_im = self.seq.augment_image(np.array(im))
    return Image.fromarray(augmented_im)


class JpgTransformer(Transformer):
  def __init__(self):
    # self.seq = iaa.Resize((75, 150))
    # self.seq = iaa.Grayscale(alpha=1.0)
    minc, maxc = 0, 0.35 # Hard
    crop_percent = ([minc, maxc], [minc, maxc], [minc, maxc], [minc, maxc])
    self.seq = iaa.Crop(percent=crop_percent, keep_size=False)
  

class CropTransformer(Transformer):
  def __init__(self):
    # minc, maxc = 0.05, 0.3 # Medium
    minc, maxc = 0, 0.35 # Hard
    # minc, maxc = 0.35, 0.35 # Hard
    crop_percent = ([minc, maxc], [minc, maxc], [minc, maxc], [minc, maxc])
    self.seq = iaa.Crop(percent=crop_percent, keep_size=False)

    # self.seq = iaa.Sequential([
    #   iaa.PadToFixedSize(width=100, height=100),
    #   iaa.CropToFixedSize(width=100, height=100)
    # ])

class RotateTransformer(Transformer):
  def __init__(self):
    rot = 180
    self.seq = iaa.Affine(rotate=(-rot, rot), fit_output=False)

class RotateCropTransformer(Transformer):
  def __init__(self):
    rot = 180
    minc, maxc = 0, 0.35 # Hard
    crop_percent = ([minc, maxc], [minc, maxc], [minc, maxc], [minc, maxc])
    self.seq = iaa.SomeOf((1, None), [
      iaa.Affine(rotate=(-rot, rot)),
      iaa.Crop(percent=crop_percent, keep_size=False),
    ], random_order=True)

class AllTransformer(Transformer):
  def __init__(self):
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    sometimes_x = lambda x, aug: iaa.Sometimes(x, aug)

    self.seq = iaa.Sequential([
      iaa.Sequential([
        sometimes(blur()),
        sometimes(noise()),
        sometimes(contrast()),
        sometimes(fuckery()),
        sometimes(rotate()),
        sometimes(geometric()),
        # sometimes(segmentation()),
        sometimes(crop()),
        sometimes_x(0.25, dropout()),
        # TODO: Blend with other images.
      ],
      random_order=True),
    ])

def blur():
  return iaa.OneOf([
    iaa.GaussianBlur((0, 3.0)),
    iaa.AverageBlur(k=(2, 7)),
    iaa.MedianBlur(k=(3, 11)),
    iaa.BilateralBlur(d=(3, 10), sigma_color=(10, 250), sigma_space=(10, 250)),
  ])

def noise():
  return iaa.OneOf([
    iaa.Add((-10, 10), per_channel=0.5),
    iaa.AddElementwise((-10, 10), per_channel=0.5),
    iaa.AdditiveGaussianNoise(scale=0.1*255, per_channel=0.5),
    iaa.AdditiveLaplaceNoise(scale=0.1*255, per_channel=0.5),
    iaa.AdditivePoissonNoise(lam=2, per_channel=0.5),
    iaa.Multiply((0.5, 1.5), per_channel=0.5)
  ])

def contrast():
  return iaa.OneOf([
    iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5),
    iaa.GammaContrast((0.5, 1.5), per_channel=0.5),
    iaa.HistogramEqualization(),
    iaa.LinearContrast((0.5, 1.5), per_channel=0.5),
    iaa.LogContrast((0.5, 1.5), per_channel=0.5),
    iaa.SigmoidContrast((5, 20), (0.25, 0.75), per_channel=0.5),
  ])

def fuckery():
  return iaa.OneOf([
    # iaa.Invert(0.1, per_channel=0.5),
    iaa.JpegCompression(compression=(80, 97)),
  ])


def dropout():
  return iaa.OneOf([
    # Low size_percent -> big dropout rectangles
    iaa.CoarseDropout((0.0, 0.5), size_percent=0.01, min_size=2),
    iaa.CoarseDropout((0.0, 0.5), size_percent=(0.01, 0.02)),
    iaa.CoarseDropout((0.0, 0.5), size_percent=(0.02, 0.1)),
  ])

def rotate():
  return iaa.SomeOf(2, [
    iaa.Fliplr(0.2),
    iaa.Flipud(0.1),
    iaa.Affine(rotate=(-180, 180))
  ])

def geometric():
  return iaa.OneOf([
    iaa.Affine(scale=2.0),
    iaa.ElasticTransformation(alpha=(0.0, 70.0), sigma=5.0),
    iaa.PerspectiveTransform(scale=(0.01, 0.10)),
    # iaa.PiecewiseAffine(scale=(0.01, 0.05)),
    # iaa.Rot90((1, 3), keep_size=False)
  ])

def segmentation():
  return iaa.Sometimes(0.2,
    iaa.Superpixels(p_replace=(0.25, 1.0), n_segments=(64, 128))
  )

def crop():
  return iaa.OneOf([
    iaa.Crop(percent=([0.05, 0.1], [0.05, 0.1], [0.05, 0.1], [0.05, 0.1])),
  ])

def uniform_size(im_size):
  return iaa.Sequential([
    iaa.PadToFixedSize(width=im_size, height=im_size),
    iaa.CropToFixedSize(width=im_size, height=im_size)
  ])

def color():
  # TODO: AddToHueAndSaturation & Grayscale. Blend augs. Convs
  return iaa.OneOf([
    iaa.Crop(percent=([0.05, 0.1], [0.05, 0.1], [0.05, 0.1], [0.05, 0.1])),
  ])


if __name__ == '__main__':
  import random
  seed = random.randint(0, 10000)
  ia.seed(seed)

  # transformer = AllTransformer()
  # transformer = JpgTransformer()
  # transformer = RotateTransformer()
  transformer = CropTransformer()
  # transformer = RotateCropTransformer()
  im = Image.open('datasets/sheepie.jpg')
  transformer.grid([np.array(im)])
