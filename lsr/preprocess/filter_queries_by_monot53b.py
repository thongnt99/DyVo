import json
import ir_datasets
qid2query = {}
with open("data/robust04/genfree/queries.tsv", "r") as f:
    for line in f:
        qid, text = line.split("\t")
        qid2query[qid] = text
qrels = json.load(open("data/robust04/genfree/qrels.json"))
scores = json.load(open("data/robust04/genfree/monot5_3b_scores.json"))
dataset = ir_datasets.load("disks45/nocr/trec-robust-2004")
good_query = 0
bad_query = 0
for qid in qid2query:
    # print(f"Query: {qid2query[qid]}")
    rel_doc_id = list(qrels[qid])[0]
    if qid in scores and scores[qid][rel_doc_id] >= 0.9:
        good_query += 1
    else:
        bad_query += 1
print(good_query)
print(bad_query)
# print(f"Relevance score: {scores[qid][rel_doc_id]}")
# print(f"Rel doc: {dataset.docs_store().get(rel_doc_id).body}")
# print("================================================")
# input()
