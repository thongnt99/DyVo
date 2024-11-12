import ir_datasets
import numpy as np
from lsr.utils.dataset_utils import read_ce_score, read_qrels
from tqdm import tqdm
import random
random.seed(42)
ce_scores = read_ce_score("hfds:sentence-transformers/msmarco-hard-negatives")
qrels = read_qrels("irds:msmarco-passage/train")
query_ids = list(qrels.keys())
with open("data/msmarco_triplets.tsv", "w") as f:
    for epoch in tqdm(range(50)):
        np.random.shuffle(query_ids)
        for q_id in query_ids:
            pos_id = random.choice(list(qrels[q_id].keys()))
            if pos_id in ce_scores[q_id]:
                neg_lists = list(ce_scores[q_id].keys())
                neg_id = pos_id
                while neg_id == pos_id:
                    neg_id = random.choice(neg_lists)
                f.write(
                    f"{q_id}\t{pos_id}\t{ce_scores[q_id][pos_id]}\t{neg_id}\t{ce_scores[q_id][neg_id]}\n")
