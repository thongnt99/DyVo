import random
from lsr.utils.dataset_utils import (
    read_collection,
    read_qrels,
    read_queries,
    read_triplets,
)
from torch.utils.data import Dataset
from collections import defaultdict


class MultipleNegatives(Dataset):
    def __init__(
        self,
        collection_path: str,
        queries_path: str,
        triplet_ids_path: str,
        train_group_size: int,
        query_fields=["text"],
        doc_fields=["text"]
    ):
        self.docs_dict = read_collection(
            collection_path, text_fields=doc_fields)
        _, self.query2pos, self.query2neg = read_triplets(triplet_ids_path)
        self.q_dict = {
            item[0]: item[1]
            for item in read_queries(queries_path, text_fields=query_fields)
            if item[0] in self.query2pos
        }
        self.qids = list(self.q_dict.keys())
        self.train_group_size = train_group_size

    def __len__(self):
        return len(self.q_dict)

    def __getitem__(self, item):
        qid = self.qids[item]
        query = self.q_dict[qid]
        pos_id = random.choice(self.query2pos[qid])
        pos_psg = self.docs_dict[pos_id]
        doc_group = []
        doc_group.append(pos_psg)
        if len(self.query2neg[qid]) < self.train_group_size - 1:
            negs = random.choices(
                self.query2neg[qid], k=self.train_group_size - 1)
        else:
            negs = random.sample(
                self.query2neg[qid], k=self.train_group_size - 1)
        doc_group.extend([self.docs_dict[neg] for neg in negs])
        return {"query_id": qid, "query_text": query, "doc_id": [pos_id] + list(negs),  "doc_text": doc_group}
