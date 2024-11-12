import json
import re
import pandas as pd
import numpy as np
import json
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
import torch


class BERTCandidate:
    def __init__(self,
                 query_ent_path="/projects/0/guse0488/dataset/entities/msmarco-passage/msmarco_queries.jsonl",
                 doc_ent_path="/projects/0/guse0488/dataset/entities/msmarco-passage/msmarco_passages.jsonl"):
        #  entity_dict="data/entity_list.json",
        #  top_k_query=20,
        #  bert_embedding_model="distilbert-base-uncased"):
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
        # self.entity_list = json.load(open(entity_dict))
        # self.tokenizer = AutoTokenizer.from_pretrained(bert_embedding_model)
        # self.emb_model = AutoModel.from_pretrained(
        #     bert_embedding_model).embeddings.word_embeddings

    def retrieve_candidates(self, text, text_id,  type="query"):
        entity_ids = []
        if type == "query":
            entity_ids = self.query_entities[text_id]
        else:
            entity_ids = self.doc_entities[text_id]
        entity_ids = [int(eid) for eid in entity_ids]
        entity_masks = [1] * len(entity_ids)
        return entity_ids, entity_masks
        # entity_names = [self.entity_list[eid] for eid in entity_ids]
        # if entity_names:
        #     entity_tokens = self.tokenizer(
        #         entity_names, padding=True, return_tensors="pt").to("cuda")
        #     token_embs = self.emb_model(
        #         entity_tokens["input_ids"]) * entity_tokens["attention_mask"].unsqueeze(-1)
        #     entity_embs = token_embs.sum(
        #         dim=1) / entity_tokens["attention_mask"].sum(dim=1).unsqueeze(-1)
        #     entity_embs = entity_embs.tolist()
        #     entity_scores = [1.0] * len(entity_ids)
        #     return [entity_ids, entity_embs, entity_scores]
        # else:
        #     return [[], [], []]
