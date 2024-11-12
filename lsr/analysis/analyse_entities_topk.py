from collections import defaultdict
import json
import ir_measures
# run_file = "./outputs/qmlp_dmlm_robust04_cocondenser_msmarco_pretrained_inparsv2_monot53b_distillation_l1_0.0_0.00001/12-18-2023/test_run.trec"
run_file = "data/robust04/inparsv2/tmp-run-robust04.txt"
# query_entities = "/projects/0/guse0488/dataset/entities/trec-robust04/queries_desc.jsonl"
query_entities = "/projects/0/guse0488/dataset/entities/trec-robust04/queries_inparsv2.jsonl"
doc_entities = "/projects/0/guse0488/dataset/entities/trec-robust04/docs.jsonl"


def read_entities(fn):
    id2entites = {}
    with open(fn, "r") as f:
        for line in f:
            row = json.loads(line.strip())
            if len(row["entities"]) > 0:
                if row["id"].startswith("inparsv2_"):
                    row['id'] = row["id"].replace("inparsv2_", "")
                id2entites[row["id"]] = set(row["entities"])
    return id2entites


qid2entities = read_entities(query_entities)
did2entities = read_entities(doc_entities)
trec_run = ir_measures.read_trec_run(run_file)
q2dscores = defaultdict(dict)
for row in trec_run:
    q2dscores[row.query_id][row.doc_id] = row.score
match_counts = [0 for _ in range(1000)]
for qid in qid2entities:
    doc_ids = q2dscores[qid].keys()
    doc_ids = sorted(
        doc_ids, key=lambda did: q2dscores[qid][did], reverse=True)
    for rank, doc_id in enumerate(doc_ids):
        if doc_id in did2entities:
            overlap = qid2entities[qid].intersection(did2entities[doc_id])
            if len(overlap) > 0:
                match_counts[rank] += 1
match_counts = [c*1.0/len(qid2entities) for c in match_counts]
json.dump(match_counts, open("data/robust04/inparsv2/entity_matches.json", "w"))
