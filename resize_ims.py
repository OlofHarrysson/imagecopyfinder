from pathlib import Path
from PIL import Image

def main(im_dir):
  for im_path in im_dir.iterdir():
    if im_path.name[0] == '.':
      continue

    im = Image.open(im_path)
    im.thumbnail((600, 600))
    save_path = str(im_path).replace('_src', '')
    im.save(save_path)

if __name__ == '__main__':
  im_dir = Path('datasets/copydays/original_src')
  # im_dir = Path('datasets/copydays/sample/original_src')
  main(im_dir)
  # im_dir = Path('datasets/copydays/sample/strong_src')
  im_dir = Path('datasets/copydays/strong_src')
  main(im_dir)
  