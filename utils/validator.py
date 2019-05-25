import torch
import torch.nn as nn
from data import TripletDataset, CopyDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image, to_tensor
import dataclasses
from dataclasses import dataclass
from collections import OrderedDict

class Validator():
  def __init__(self, model, logger, config):
    self.model, self.logger, self.config = model, logger, config

    def collate(batch):
      return batch[0]

    # self.dataset = CopyDataset('datasets/copydays/sample/index.json')
    self.dataset = CopyDataset('datasets/copydays/index.json')
    self.dataloader = DataLoader(self.dataset, batch_size=1, collate_fn=collate, num_workers=config.num_workers)

  def validate(self, step):
    print("~~~~~~~~ Started Validation ~~~~~~~~")
    self.model.eval()

    # Create embeddings
    query_embeddings = OrderedDict()
    database_embeddings = OrderedDict()

    for ind, data in enumerate(self.dataloader, 1):
      if ind > self.config.max_val_batches:
        break

      im, im_type, match_id, im_id = data
      entry = Entry(im_type, match_id, im_id)

      inp = to_tensor(im).unsqueeze(0)
      with torch.no_grad():
        outp = self.model.predict_embedding(inp).cpu() # TODO: GPU?

      if im_type == 'query':
        query_embeddings[entry] = outp
      elif im_type == 'database':
        database_embeddings[entry] = outp

    # Calculates number of corrects
    best_matches, ranks = [], []
    database_keys = list(database_embeddings.keys())
    is_match = lambda q, db: q.match_id == db.match_id
    for query_entry, q_emb in query_embeddings.items():

      # Query & database entry distances
      distances = self.calc_distance(q_emb, database_embeddings)
      # distances = self.model.calc_distance(q_emb, database_embeddings)

      # Finds best match & rank of the prediction
      _, dist_sorted = distances.topk(distances.size(0), largest=False)
      for rank_number, dist_ind in enumerate(dist_sorted, 1):
        db_entry = database_keys[dist_ind]
        
        if rank_number == 1: # Best match
          best_matches.append((query_entry.im_id, db_entry.im_id))

        if is_match(query_entry, db_entry): # Rank
          ranks.append(rank_number)
          break

    self.logger.log_rank(ranks, step)
    self.logger.log_accuracy(ranks, step)
    # self.save_matches(best_matches)
    self.model.train()
    print("~~~~~~~~ Finished Validation ~~~~~~~~")

  def calc_distance(self, query, database):
    ''' Returns distances, an 1-dim tensor for query to all database
        Returns match_ids, a 1-dim list for database ind
    '''
    dist = nn.PairwiseDistance(p=1)
    distances = torch.tensor([])
    for db_entries, db_emb in database.items():
      dd = self.model.calc_distance(query, db_emb).squeeze(dim=0)
      # dd = dist(query, db_emb)
      distances = torch.cat((distances, dd))
      
    return distances

  def save_matches(self, im_id_matches):
    ''' Save best matches as a concatenated image '''
    for query_id, db_id in im_id_matches:
      query_im, *_ = self.dataset[query_id]
      db_im, *_ = self.dataset[db_id]

      min_size = min(min(query_im.size), min(db_im.size))
      query_im.thumbnail((min_size, min_size))
      db_im.thumbnail((min_size, min_size))
      
      query_im = squarify(query_im)
      db_im = squarify(db_im)
      im = torch.cat((query_im, db_im), dim=2)

      concat_im = to_pil_image(im)
      concat_im.save('output/%s.png' % query_id)


  

def squarify(im):
  im = to_tensor(im)
  c, h, w = im.size()
  if h > w:
    padding = (h-w, 0, 0, 0)
  else:
    padding = (0, 0, w-h, 0)
  return F.pad(im, padding)

@dataclass(frozen=True, eq=True)
class Entry:
  im_type: str
  match_id: int
  im_id: int