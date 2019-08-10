import torchvision.transforms as transforms
from PIL import Image
import imgaug as ia
import imgaug.augmenters as iaa
# from imgaug import parameters as iap
import numpy as np

class FlipTransformer():
  def __init__(self):
    self.seq = iaa.SomeOf((1, None), [
      iaa.Fliplr(1.0),
      iaa.Rot90((1, 3), keep_size=False)
    ], random_order=True)

  def __call__(self, im):
    augmented_im = self.seq.augment_image(np.array(im))
    return Image.fromarray(augmented_im)

  def numpy_transform(self, im):
    return self.seq.augment_image(np.array(im))

class CropTransformer():
  def __init__(self):
    # minc, maxc = 0.05, 0.3 # Medium
    minc, maxc = 0.25, 0.4 # Hard
    crop_percent = ([minc, maxc], [minc, maxc], [minc, maxc], [minc, maxc])
    self.seq = iaa.Crop(percent=crop_percent, keep_size=False)

  def __call__(self, im):
    augmented_im = self.seq.augment_image(np.array(im))
    return Image.fromarray(augmented_im)

  def numpy_transform(self, im):
    return self.seq.augment_image(np.array(im))


class Transformer():
  def __init__(self):
    self.im_size = 100 # TODO: When do I actually want to do this? Also resize in the dataset function. I'd say do crop in the dropout function since they don't match that well
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    self.seq = iaa.Sequential([
      iaa.Sequential([
        sometimes(blur()),
        sometimes(noise()),
        sometimes(contrast()),
        sometimes(fuckery()),
        sometimes(flip()),
        sometimes(geometric()),
        sometimes(segmentation()),
        sometimes(crop()),
        sometimes(iaa.OneOf([
          dropout(),
          # uniform_size(self.im_size), TODO: this???
        ])),
        # TODO: Blend with other images.
      ],
      random_order=True),
      
    ])

  def __call__(self, im):
    augmented_im = self.seq.augment_image(np.array(im))
    return Image.fromarray(augmented_im)

  def numpy_transform(self, im):
    return self.seq.augment_image(np.array(im))

  def grid(self, im):
    self.seq.show_grid(im, cols=6, rows=4)


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
    iaa.Invert(0.1, per_channel=0.5),
    iaa.JpegCompression(compression=(80, 95)),
  ])


def dropout():
  return iaa.OneOf([
    # Low size_percent -> big dropout rectangles
    iaa.CoarseDropout((0.0, 0.5), size_percent=0.01, min_size=2),
    iaa.CoarseDropout((0.0, 0.5), size_percent=(0.01, 0.02)),
    iaa.CoarseDropout((0.0, 0.5), size_percent=(0.02, 0.1)),
  ])

def flip():
  return iaa.SomeOf(2, [
    iaa.Fliplr(0.2),
    iaa.Flipud(0.1),
  ])

def geometric():
  return iaa.OneOf([
    iaa.Affine(scale=2.0),
    iaa.ElasticTransformation(alpha=(0.0, 70.0), sigma=5.0),
    iaa.PerspectiveTransform(scale=(0.01, 0.10)),
    # iaa.PiecewiseAffine(scale=(0.01, 0.05)),
    iaa.Rot90((1, 3), keep_size=False)
  ])

def segmentation():
  return iaa.Sometimes(0.2,
    iaa.Superpixels(p_replace=(0.25, 1.0), n_segments=(16, 128))
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

if __name__ == '__main__':
  import random
  seed = random.randint(0, 10000)
  ia.seed(seed)

  # transformer = Transformer()
  transformer = FlipTransformer()
  im = Image.open('datasets/sheepie.jpg')
  transformer.grid([np.array(im)])
