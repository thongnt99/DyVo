import argparse
import json
parser = argparse.ArgumentParser()
parser.add_argument("--q", type=str, default="data/robust04/queries.tsv")
parser.add_argument("--q_ent", type=str,
                    default="data/robust04/queries_ent_cand_dpr.jsonl")
parser.add_argument("--ent_dict", type=str, default="data/entity_list.json")

args = parser.parse_args()
qid2text = {}
with open(args.q, "r") as f:
    for line in f:
        qid, qtext = line.strip().split("\t")
        qid2text[qid] = qtext
entities = json.load(open(args.ent_dict))
with open(args.q_ent, "r") as f:
    for line in f:
        q = json.loads(line)
        print(q["id"])
        print(qid2text[str(q["id"])])
        print([entities[eid] for eid in q["entities"][:10]])
        input()
