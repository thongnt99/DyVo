import ir_datasets
from collections import defaultdict
import json

rel_query_entity_file = "/projects/0/guse0488/dataset/entities/msmarco-passage/msmarco_queries.jsonl"
rel_doc_entity_file = "/projects/0/guse0488/dataset/entities/msmarco-passage/msmarco_passages.jsonl"
query_entity_output_file = "/projects/0/guse0488/dataset/entities/msmarco-passage/msmarco_queries_retrieval.jsonl"
doc_entity_output_file = "/projects/0/guse0488/dataset/entities/msmarco-passage/msmarco_passages_retrieval.jsonl"
msmarco_train = ir_datasets.load("msmarco-passage/train")
msmarco_dev = ir_datasets.load("msmarco-passage/dev/small")
q2reldocs = defaultdict(list)
d2queries = defaultdict(list)
for qrel in msmarco_train.qrels_iter():
    if qrel.relevance > 0:
        q2reldocs[qrel.query_id].append(qrel.doc_id)
        d2queries[qrel.doc_id].append(qrel.query_id)

for qrel in msmarco_dev.qrels_iter():
    if qrel.relevance > 0:
        q2reldocs[qrel.query_id].append(qrel.doc_id)
        d2queries[qrel.doc_id].append(qrel.query_id)
q_linked_entities = defaultdict(list)
with open(rel_query_entity_file, "r") as fin:
    for line in fin:
        query = json.loads(line)
        q_linked_entities[str(query["id"])] = query["entities"]
d_linked_entities = defaultdict(list)
with open(rel_doc_entity_file, "r") as fin:
    for line in fin:
        doc = json.loads(line)
        d_linked_entities[str(doc["id"])] = doc["entities"]
with open(query_entity_output_file, "w") as fout:
    for q_id in q_linked_entities:
        q_rel_entities = q_linked_entities[q_id]
        for rel_did in q2reldocs[q_id]:
            q_rel_entities.extend(d_linked_entities[rel_did])
        q_rel_entities = list(set(q_rel_entities))
        json_q = {"id": q_id, "entities": q_rel_entities}
        fout.write(json.dumps(json_q)+"\n")

with open(doc_entity_output_file, "w") as fout:
    for d_id in d_linked_entities:
        d_rel_entities = d_linked_entities[d_id]
        for q_id in d2queries[d_id]:
            d_rel_entities.extend(q_linked_entities[q_id])
        d_rel_entities = list(set(d_rel_entities))
        json_d = {"id": d_id, "entities": d_rel_entities}
        fout.write(json.dumps(json_d)+"\n")
