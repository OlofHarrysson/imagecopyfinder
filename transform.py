import torchvision.transforms as transforms
from PIL import Image

class Transformer():
  def __init__(self):
    self.transform = transforms.Compose([
      transforms.ColorJitter(0.5, 0.5, 0.5, 0.5),
      transforms.RandomHorizontalFlip(p=0.5),
      # transforms.RandomResizedCrop(200, (0.75, 1)),
      # transforms.RandomResizedCrop(32, (0.75, 1)),
      # transforms.RandomRotation(10),
      transforms.RandomVerticalFlip(p=0.5),
      transforms.RandomAffine((-30, 30), (0.2, 0.2), shear=30),
    ])

  def __call__(self, im):
    return self.transform(im)

if __name__ == '__main__':
  transformer = Transformer()
  im = Image.open('datasets/sheepie.jpg')

  t_im = transformer(im)
  print(t_im)
  t_im.show()