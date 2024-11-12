from tqdm import tqdm
from collections import defaultdict
import json
import ir_measures
import ir_datasets
from ir_measures import *
import argparse
parser = argparse.ArgumentParser("Argument Parser")
parser.add_argument("--query_cand", type=str,
                    default="/projects/0/guse0488/dataset/entities/trec-robust04/queries_desc.jsonl")
parser.add_argument("--doc_cand", type=str, default="/projects/0/guse0488/dataset/entities/trec-robust04/docs.jsonl"
                    )
args = parser.parse_args()


def read_entities(fn):
    id2entites = defaultdict(set)
    with open(fn, "r") as f:
        for line in tqdm(f, desc="Reading entities"):
            row = json.loads(line.strip())
            id2entites[row["id"]] = set(row["entities"])
    return id2entites


qid2entities = read_entities(args.query_cand)
# avg_entities_per_query = sum(
#     [len(k) for k in qid2entities.keys()])/len(qid2entities)
# print(f"Average number of entities per query: {avg_entities_per_query}")
did2entities = read_entities(args.doc_cand)
# avg_entities_per_doc = sum(
#     [len(k) for k in did2entities.keys()])/len(did2entities)
# print(f"Average number of entities per doc: {avg_entities_per_doc}")
dataset = ir_datasets.load("disks45/nocr/trec-robust-2004")
q2dscores = defaultdict(dict)
# for qid in tqdm(qid2entities, "Calculating scores"):
#     for did in did2entities:
#         overlap = len(
#             qid2entities[qid].intersection(did2entities[did]))*1.0
#         if overlap > 0:
#             q2dscores[qid][did] = overlap
for row in tqdm(dataset.qrels_iter(), desc="Calcualting scores"):
    if row.query_id in qid2entities and row.doc_id in did2entities:
        overlap = len(qid2entities[row.query_id].intersection(
            did2entities[row.doc_id]))*1.0
        if overlap > 0:
            q2dscores[row.query_id][row.doc_id] = row.relevance
upperboud_metrics = ir_measures.calc_aggregate(
    [NDCG@10, NDCG@20, R@100, R@1000, R@2000, R@10000, R@20000, R@50000, R@100000, R@500000], dataset.qrels_iter(), q2dscores)
print(upperboud_metrics)
