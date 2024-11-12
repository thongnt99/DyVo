import json
from torch.utils.data import Dataset
import ir_datasets
from tqdm import tqdm
import gzip
import pickle
import random

from lsr.utils.dataset_utils import (
    read_collection,
    read_queries,
    read_qrels,
    read_ce_score,
    read_entity_annotations
)


def mask(query, doc, p):
    qwords = set(query.split(" "))
    dwords = doc.split(" ")
    for i in range(len(dwords)):
        if dwords[i] in qwords:
            if random.random() < p:
                dwords[i] = '[MASK]'
    doc = " ".join(dwords)
    return doc


class TripletIDDistilDataset(Dataset):
    """
    Dataset with teacher's scores for distillation
    """

    def __init__(
        self,
        collection_path: str,
        queries_path: str,
        qrels_path: str,
        ce_score_dict: str,
        train_group_size=2,
        doc_fields: list = ["text"],
        mask_query: float = 0.0,
        neg_as_pos: bool = True,
    ):
        super().__init__()
        self.doc_dict = read_collection(
            collection_path, text_fields=doc_fields)
        self.queries = read_queries(queries_path)
        self.qrels = read_qrels(qrels_path)
        self.ce_score = read_ce_score(ce_score_dict)
        self.queries = [
            pair for pair in self.queries if pair[0] in self.ce_score]
        self.train_group_size = train_group_size
        self.mask_query = mask_query
        self.neg_as_pos = neg_as_pos

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        q_id, q_text = self.queries[idx]
        if q_id in self.qrels:
            doc1_id = random.choice(list(self.qrels[q_id].keys()))
            # IMPORTANT: negative as positive
            if self.neg_as_pos and random.random() >= 0.6:
                doc1_id = random.choice(list(self.ce_score[q_id].keys()))
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
        # if self.mask_query > 0:
        #     doc_list = [mask(q_text, doc_text, p=self.mask_query)
        #                 for doc_text in doc_list]
        score_list.extend([self.ce_score[q_id][doc_id]
                          for doc_id in neg_doc_ids])
        return {"query_id": q_id, "query_text": q_text, "doc_id": [doc1_id] + neg_doc_ids,  "doc_text": doc_list, "score": score_list}
