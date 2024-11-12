from collections import defaultdict
import json

query_entity_path = "/projects/0/guse0488/dataset/entities/trec-robust04/queries_desc_inparsv2.jsonl"
doc_entity_path = "/projects/0/guse0488/dataset/entities/trec-robust04/docs.jsonl"
queries_path = 'data/robust04/inparsv2/prefix_topics-robust04.tsv'
qrels_path = 'data/robust04/inparsv2/prefix_robust04_qrels.json'
ce_score_dict = 'data/robust04/inparsv2/prefix_monot5_3b_scores.json'

queries = {}
with open(queries_path, "r") as f:
    for line in f:
        qid, qtext = line.strip().split("\t")
        queries[qid] = qtext


def read_entities(fp):
    d = defaultdict(set)
    with open(fp, "r") as f:
        for line in f:
            row = json.loads(line.strip())
            d[row["id"]] = set(row["entities"])
    return d


query_entities = read_entities(query_entity_path)
doc_entities = read_entities(doc_entity_path)
monot5_scores = json.load(open(ce_score_dict, "r"))

qrels = json.load(open(qrels_path, "r"))
monot5_scores_with_entity_match = defaultdict(dict)
for qid in monot5_scores:
    for did in monot5_scores[qid]:
        if len(query_entities[qid].intersection(doc_entities[did])) > 0:
            monot5_scores_with_entity_match[qid][did] = monot5_scores[qid][did]
json.dump(monot5_scores_with_entity_match, open(
    "data/robust04/inparsv2/entity_monot5_3b_scores.json", "w"))
entity_qrels = defaultdict(dict)
for qid in qrels:
    for did in qrels[qid]:
        if qid in monot5_scores_with_entity_match and did in monot5_scores_with_entity_match[qid]:
            entity_qrels[qid][did] = qrels[qid][did]
json.dump(entity_qrels, open(
    "data/robust04/inparsv2/entity_robust04_qrels.json", "w"))

with open("data/robust04/inparsv2/entity_topics-robust04.tsv", "w") as f:
    for qid in queries:
        if qid in entity_qrels:
            f.write(f"{qid}\t{queries[qid]}\n")
