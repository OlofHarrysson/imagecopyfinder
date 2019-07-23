import torch
import torch.nn as nn
from data import TripletDataset, CopyDataset, OnlineTransformDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image, to_tensor
import dataclasses
from dataclasses import dataclass
from collections import OrderedDict, defaultdict

class Validator():
  def __init__(self, model, logger, config):
    self.model, self.logger, self.config = model, logger, config

    def collate(batch):
      return batch[0]

    index_file = f'{config.validation_dataset}/index.json'
    # self.dataset = CopyDataset(index_file, config)
    self.dataset = OnlineTransformDataset(config.validation_dataset)
    self.dataloader = DataLoader(self.dataset, batch_size=1, collate_fn=collate, num_workers=config.num_workers)

  def validate(self, step):
    self.model.eval()
    query_embeddings, database_embeddings = self.calc_embeddings()

    # Calculates number of corrects
    ranks_dict = defaultdict(list)
    database_keys = list(database_embeddings.keys())
    is_match = lambda q, db: q.match_id == db.match_id
    for query_entry, q_emb in query_embeddings.items():

      # Query & database entry similarities
      similarity_dict = self.model.similarities(q_emb, database_embeddings)

      for metric_name, similarities in similarity_dict.items():

        # Finds best match & rank of the prediction
        _, similarities_sorted = similarities.topk(similarities.size(0))
        for rank_number, sim_ind in enumerate(similarities_sorted, 1):
          db_entry = database_keys[sim_ind]
          
          if is_match(query_entry, db_entry): # Rank
            ranks_dict[metric_name].append(rank_number)
            break

    for metric_name, ranks in ranks_dict.items():
      self.logger.log_rank(ranks, step, metric_name)
      self.logger.log_accuracy(ranks, step, metric_name)

    self.model.train()

  def calc_embeddings(self):
    query_embeddings, database_embeddings = OrderedDict(), OrderedDict()

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

    return query_embeddings, database_embeddings

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