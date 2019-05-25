from pathlib import Path
from PIL import Image, ImageDraw
import random

def main():
  is_hidden_file = lambda path: path.name[0] == '.'
  im_paths = Path('datasets/cifar_sample').iterdir()
  
  for path in im_paths:
    if is_hidden_file(path):
      continue
    im = Image.open(path)
    draw = ImageDraw.Draw(im)

    im_size = 32
    y1 = random.randint(0, im_size//2)
    x1 = random.randint(0, im_size//2)

    min_size = 5
    h = random.randint(min_size, im_size//2)
    w = random.randint(min_size, im_size//2)

    y2 = y1 + h
    x2 = x1 + w
    assert y2 <= im_size and x2 <= im_size

    color = random.randint(0, 256)
    draw.rectangle([x1, y1, x2, y2], fill=color)

    # draw.line((0, 0) + im.size, fill=128)
    # draw.line((0, im.size[1], im.size[0], 0), fill=128)

    im.save('datasets/color_changed/%s.png' % path.stem)

if __name__ == '__main__':
  main()