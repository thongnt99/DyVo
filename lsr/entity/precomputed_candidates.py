import json
import re
import pandas as pd
import numpy as np
import json
from collections import defaultdict


class PrecomputedCandidate:
    def __init__(self,
                 query_ent_path="/projects/0/guse0488/dataset/entities/msmarco-passage/msmarco_queries.jsonl",
                 doc_ent_path="/projects/0/guse0488/dataset/entities/msmarco-passage/msmarco_passages.jsonl",
                 emb_path="data/wiki_embs.pkl.npy"):
        self.query_entities = defaultdict(list)
        self.doc_entities = defaultdict(list)
        with open(query_ent_path, "r") as f:
            for line in f:
                query = json.loads(line.strip())
                query_id = str(query["id"])
                query_entities = list(set(query["entities"]))
                self.query_entities[query_id] = query_entities

        with open(doc_ent_path, "r") as f:
            for line in f:
                doc = json.loads(line.strip())
                doc_id = str(doc["id"])
                doc_entities = list(set(doc["entities"]))
                self.doc_entities[doc_id] = doc_entities
        self.embs = np.load(emb_path)

    def retrieve_candidates(self, text, text_id,  type="query"):
        entity_ids = []
        if type == "query":
            entity_ids = self.query_entities[text_id]
        else:
            entity_ids = self.doc_entities[text_id]
        entity_ids = [int(eid) for eid in entity_ids]
        entity_embs = [self.embs[ent_id].tolist()
                       for ent_id in entity_ids]
        entity_scores = [1.0] * len(entity_ids)
        return [entity_ids, entity_embs, entity_scores]
