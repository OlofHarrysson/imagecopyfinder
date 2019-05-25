import torchvision.transforms as transforms
from PIL import Image
import imgaug as ia
import imgaug.augmenters as iaa
# from imgaug import parameters as iap
import numpy as np

class Transformer():
  def __init__(self):
    # self.transform = transforms.Compose([
    #   transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
    #   transforms.RandomHorizontalFlip(p=0.5),
    #   # transforms.RandomResizedCrop(200, (0.75, 1)),
    #   # transforms.RandomResizedCrop(32, (0.75, 1)),
    #   # transforms.RandomRotation(10),
    #   transforms.RandomVerticalFlip(p=0.5),
    #   transforms.RandomAffine((-30, 30), (0.2, 0.2), shear=30),
    # ])

    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    self.seq = iaa.Sequential(
      [
        blur(),
        noise(),
        contrast(),
        fuckery(),
        dropout(),
      ],
      random_order=True
    )

  def __call__(self, im):
    # return self.transform(im)
    augmented_im = self.seq.augment_image(np.array(im))
    return Image.fromarray(augmented_im)

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
    iaa.JpegCompression(compression=(20, 95)),
  ])


def dropout():
  return iaa.OneOf([
    iaa.CoarseDropout((0.0, 0.5), size_percent=(0.02, 0.5))
    # TODO: Coursesaltpepper?
  ])


if __name__ == '__main__':
  import psutil

  def clear_envs(viz):
    [viz.close(env=env) for env in viz.get_env_list()] # Kills wind

  def kill_image_window():
    for proc in psutil.process_iter():
      if proc.name() == "Preview":
          proc.kill()
          # print(proc)

  transformer = Transformer()
  im = Image.open('datasets/sheepie.jpg')
  viz = visdom.Visdom(port='6006')
  clear_envs(viz)


  for i in range(3):
    t_im = transformer(im)
    t_im.show()
    input("PRESS KEY TO CONTINUE.")
    kill_image_window()