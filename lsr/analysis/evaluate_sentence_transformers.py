from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
import ir_datasets
import ir_measures
from ir_measures import *
import torch
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str,
                    default="disks45/nocr/trec-robust-2004")
args = parser.parse_args()

queries = []
query_ids = []
doc_ids = []
docs = []

dataset = ir_datasets.load(args.dataset)
for query in dataset.queries_iter():
    query_ids.append(query.query_id)
    queries.append(query.description)

for doc in dataset.docs_iter():
    doc_ids.append(doc.doc_id)
    docs.append(str(doc.title) + " " + doc.body)

model = SentenceTransformer('sentence-transformers/msmarco-distilbert-dot-v5')

with torch.cuda.amp.autocast():
    query_embs = model.encode(queries, show_progress_bar=True,
                              batch_size=128, convert_to_tensor=True)
    doc_embs = model.encode(docs, show_progress_bar=True,
                            batch_size=128, convert_to_tensor=True)

scores = query_embs @ doc_embs.T

topk_scores, topk_indices = scores.topk(k=1000, dim=1)


run = {}
for idx, q_id in enumerate(query_ids):
    topdoc_ids = [doc_ids[i] for i in topk_indices[idx].tolist()]
    run[q_id] = dict(zip(topdoc_ids, topk_scores[idx].tolist()))

qrels = list(dataset.qrels_iter())

metrics = ir_measures.calc_aggregate(
    [NDCG@10, NDCG@20, R@100, R@1000], qrels, run)
print(metrics)
