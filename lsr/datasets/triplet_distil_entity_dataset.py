import json
from torch.utils.data import Dataset
import ir_datasets
from tqdm import tqdm
import gzip
import pickle
import torch
import random

from lsr.utils.dataset_utils import (
    read_collection,
    read_queries,
    read_qrels,
    read_ce_score,
    read_entity_annotations,
    read_entity_embeddings
)


class TripletIDEntityDistilDataset(Dataset):
    """
    Dataset with teacher's scores for distillation
    """

    def __init__(
        self,
        collection_path: str,
        queries_path: str,
        entity_emb_path: int,
        query_entity_dict: str,
        document_entity_dict: str,
        qrels_path: str,
        ce_score_dict: str,
        train_group_size=2,
    ):
        super().__init__()
        self.doc_dict = read_collection(collection_path)
        self.queries = read_queries(queries_path)
        self.qrels = read_qrels(qrels_path)
        self.ce_score = read_ce_score(ce_score_dict)
        self.train_group_size = train_group_size
        self.entity2id, self.entity2emb = read_entity_embeddings(
            entity_emb_path)
        self.query_entity_dict = read_entity_annotations(
            query_entity_dict, "qid", "links", filter=lambda x: not x in self.entity2id)
        self.doc_entity_dict = read_entity_annotations(
            document_entity_dict, "pid", "passage", filter=lambda x: not x in self.entity2id)

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        q_id, q_text = self.queries[idx]
        if q_id in self.qrels:
            doc1_id = random.choice(self.qrels[q_id])
        else:
            doc1_id = random.choice(list(self.ce_score[q_id].keys()))
        doc_list = [self.doc_dict[doc1_id]]
        score_list = [self.ce_score[q_id][doc1_id]]
        all_doc_ids = list(self.ce_score[q_id].keys())
        if len(all_doc_ids) < self.train_group_size - 1:
            neg_doc_ids = random.choices(
                all_doc_ids, k=self.train_group_size - 1)
        else:
            neg_doc_ids = random.sample(
                all_doc_ids, k=self.train_group_size - 1)
        doc_list.extend([self.doc_dict[doc_id] for doc_id in neg_doc_ids])
        score_list.extend([self.ce_score[q_id][doc_id]
                          for doc_id in neg_doc_ids])
        # process query entities
        query_entity_names, query_entity_scores = self.query_entity_dict[q_id]
        # filter out entities not in our dictionary
        query_entity_ids = [self.entity2id[e_name]
                            for e_name in query_entity_names]
        query_entity_embs = [self.entity2emb[e_name]
                             for e_name in query_entity_names]
        # process doc entities
        doc_entity_names, doc_entity_scores = list(zip(*[self.doc_entity_dict[d_id]
                                                         for d_id in ([doc1_id] + neg_doc_ids)]))

        doc_entity_ids = []
        doc_entity_embs = []
        for per_doc in doc_entity_names:
            doc_entity_ids.append([self.entity2id[e_name]
                                  for e_name in per_doc])
            doc_entity_embs.append([self.entity2emb[e_name]
                                    for e_name in per_doc])

        output_dict = {"query_text": q_text, "query_entity_ids": query_entity_ids,
                       "query_entity_probs": query_entity_scores, "query_entity_embeddings": query_entity_embs,
                       "doc_text": doc_list, "doc_entity_ids": doc_entity_ids,
                       "doc_entity_probs": doc_entity_scores, "doc_entity_embeddings": doc_entity_embs, "score": score_list}
        return output_dict
