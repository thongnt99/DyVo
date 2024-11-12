import json
import ir_datasets
import numpy as np
from tqdm import tqdm
from collections import defaultdict
dataset = ir_datasets.load("msmarco-passage/train/triples-small")
query_ent_path = "/projects/0/guse0488/dataset/entities/msmarco-passage/msmarco_queries.jsonl"
doc_ent_path = "/projects/0/guse0488/dataset/entities/msmarco-passage/msmarco_passages.jsonl"
query_entities = defaultdict(list)
passage_entities = defaultdict(list)
with open(query_ent_path, "r") as f:
    for line in tqdm(f):
        ent = json.loads(line.strip())
        query_entities[ent["id"]] = ent["entities"]
with open(doc_ent_path, "r") as f:
    for line in tqdm(f):
        ent = json.loads(line.strip())
        passage_entities[str(ent["id"])] = ent["entities"]

pos_overlap = []
neg_overlap = []
for triplet in tqdm(dataset.docpairs_iter()):
    q_entities = set(query_entities[triplet.query_id])
    p_entities = set(passage_entities[triplet.doc_id_a])
    n_entities = set(passage_entities[triplet.doc_id_b])
    pos_overlap.append(len(q_entities.intersection(p_entities)))
    neg_overlap.append(len(q_entities.intersection(n_entities)))
    # print(triplet.query_id)
    # print(triplet.doc_id_a)
    # print(triplet.doc_id_b)
    # print(q_entities)
    # print(p_entities)
    # print(n_entities)
    # print(pos_overlap[-1])
    # print(neg_overlap[-1])
    # input()


pos_overlap = np.array(pos_overlap)
pos_zero = (pos_overlap == 0).sum()
pos_percentage = pos_zero/len(pos_overlap)
neg_overlap = np.array(neg_overlap)
neg_zero = (neg_overlap == 0).sum()
neg_percentage = neg_zero/len(neg_overlap)
pos_overlap_avg = np.mean(pos_overlap)
neg_overlap_avg = np.mean(neg_overlap)
print(
    f"q-pos entity overlap avg: {pos_overlap_avg}, no-overlap: {pos_zero}/{len(pos_overlap)} = {pos_percentage}")
print(
    f"q-neg entity overlap avg: {neg_overlap_avg}, no-overlap: {neg_zero}/{len(neg_overlap)} = {neg_percentage}")
