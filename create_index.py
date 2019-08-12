import dataclasses
from dataclasses import dataclass
import json
from pathlib import Path

@dataclass
class Entry:
  path: str
  im_type: str
  match_id: int
  im_id: int

def copydays_format(dataset_root, query_dir, database_dir):
  is_hidden_file = lambda path: path.name[0] == '.'

  query_paths = [f for f in query_dir.iterdir() if not is_hidden_file(f)]
  database_paths = [f for f in database_dir.iterdir() if not is_hidden_file(f)]

  file_matches = lambda f1, f2: f1[:4] == f2[:4]

  index = []
  im_id = 0
  def add_entry(item):
    index.append(dataclasses.asdict(item))
    nonlocal im_id
    im_id += 1

  for match_id, db_path in enumerate(database_paths):
    db_name = db_path.stem
    im_path = str(db_path.relative_to(dataset_root))
    entry = Entry(im_path, 'database', match_id, im_id)
    add_entry(entry)

    # Find all query images that match database image
    for query_path in query_paths:
      q_name = query_path.stem
      if file_matches(db_name, q_name):
        im_path = str(query_path.relative_to(dataset_root))
        entry = Entry(im_path, 'query', match_id, im_id)
        add_entry(entry)

  return index


def palces365_format(dataset_root, query_dir, database_dir):
  is_hidden_file = lambda path: path.name[0] == '.'

  query_paths = [f for f in query_dir.iterdir() if not is_hidden_file(f)]
  database_paths = [f for f in database_dir.iterdir() if not is_hidden_file(f)]

  file_matches = lambda f1, f2: f1 == f2

  index = []
  im_id = 0
  def add_entry(item):
    index.append(dataclasses.asdict(item))
    nonlocal im_id
    im_id += 1

  for match_id, db_path in enumerate(database_paths):
    db_name = db_path.stem
    im_path = str(db_path.relative_to(dataset_root))
    entry = Entry(im_path, 'database', match_id, im_id)
    add_entry(entry)

    # Find all query images that match database image
    for query_path in query_paths:
      q_name = query_path.stem
      if file_matches(db_name, q_name):
        im_path = str(query_path.relative_to(dataset_root))
        entry = Entry(im_path, 'query', match_id, im_id)
        add_entry(entry)

  return index


def videopairs_format(dataset_root, query_dir, database_dir):
  is_hidden_file = lambda path: path.name[0] == '.'

  query_paths = [f for f in query_dir.iterdir() if not is_hidden_file(f)]
  database_paths = [f for f in database_dir.iterdir() if not is_hidden_file(f)]

  file_matches = lambda f1, f2: f1.split('_')[0] == f2.split('_')[0]

  index = []
  im_id = 0
  def add_entry(item):
    index.append(dataclasses.asdict(item))
    nonlocal im_id
    im_id += 1

  for match_id, db_path in enumerate(database_paths):
    db_name = db_path.stem
    im_path = str(db_path.relative_to(dataset_root))
    entry = Entry(im_path, 'database', match_id, im_id)
    add_entry(entry)

    # Find all query images that match database image
    for query_path in query_paths:
      q_name = query_path.stem
      if file_matches(db_name, q_name):
        im_path = str(query_path.relative_to(dataset_root))
        entry = Entry(im_path, 'query', match_id, im_id)
        add_entry(entry)

  return index

def copydays():
  dataset_root = Path('datasets/copydays')
  query_dir = dataset_root / 'strong'
  database_dir = dataset_root / 'original'

  index = copydays_format(dataset_root, query_dir, database_dir)
  index_path = str(dataset_root / 'index.json')
  return index, index_path

def copydays_crop():
  dataset_root = Path('datasets/copydays_crop')
  query_dir = dataset_root / 'fake'
  database_dir = dataset_root / 'original'

  index = copydays_format(dataset_root, query_dir, database_dir)
  index_path = str(dataset_root / 'index.json')
  return index, index_path


def places365_val():
  dataset_root = Path('datasets/places365/validation_val')
  query_dir = dataset_root / 'fake'
  database_dir = dataset_root / 'original'

  index = palces365_format(dataset_root, query_dir, database_dir)
  index_path = str(dataset_root / 'index.json')
  return index, index_path

def videopairs():
  dataset_root = Path('datasets/videopairs')
  query_dir = dataset_root / 'queries'
  database_dir = dataset_root / 'database'

  index = videopairs_format(dataset_root, query_dir, database_dir)
  index_path = str(dataset_root / 'index.json')
  return index, index_path

def main():
  # index, index_path = copydays()
  # index, index_path = copydays_crop()
  # index, index_path = places365_val()
  index, index_path = videopairs()
  
  with open(index_path, 'w') as outfile:
    json.dump(index, outfile, indent=2)

if __name__ == '__main__':
  main()