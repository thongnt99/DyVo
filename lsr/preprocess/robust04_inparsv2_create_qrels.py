from collections import defaultdict
import json
inp_path = "data/robust04/inparsv2/triplet_ids.tsv"
q2rel = defaultdict(set)
with open(inp_path, "r") as f:
    for line in f:
        qid, pos_id, _neg_id = line.strip().split("\t")
        q2rel[qid].add(pos_id)
for qid in q2rel:
    q2rel[qid] = {doc_id: 1 for doc_id in q2rel[qid]}
json.dump(q2rel, open("data/robust04/inparsv2/robust04_qrels.json", "w"))
