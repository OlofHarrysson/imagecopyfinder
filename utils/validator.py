import torch
import torch.nn as nn
from data import setup_valdata
import dataclasses
from collections import OrderedDict, defaultdict
from models.dataclasses import Entry

class Validator():
  def __init__(self, model, logger, config, transformer):
    self.model, self.logger, self.config = model, logger, config
    self.dataloader = setup_valdata(config, transformer)
    

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
      # Check whichever rank isn't 1 and log those data.
      self.logger.log_rank(ranks, step, metric_name)
      self.logger.log_accuracy(ranks, step, metric_name)

    self.model.save(f'saved/models/{step}_model.pth')
    self.model.train()

  def calc_embeddings(self):
    query_embeddings, database_embeddings = OrderedDict(), OrderedDict()

    for ind, data in enumerate(self.dataloader, 1):
      if ind > self.config.max_val_batches:
        break

      im, im_type, match_id, im_id = data
      entry = Entry(im_type, match_id, im_id)

      with torch.no_grad():
        outp = self.model.predict_embedding(im).cpu() # TODO: GPU?

      if im_type == 'query':
        query_embeddings[entry] = outp
      elif im_type == 'database':
        database_embeddings[entry] = outp

    return query_embeddings, database_embeddings