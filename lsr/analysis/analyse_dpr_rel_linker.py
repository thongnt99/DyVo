import json
rel_linked_entities_path = "/projects/0/guse0488/dataset/entities/trec-robust04/queries_desc.jsonl"
dpr_nq_entities_path = "data/robust04/queries_ent_cand_blink.jsonl"


def read(fp):
    data = {}
    with open(fp) as f:
        for line in f:
            query = json.loads(line)
            data[str(query["id"])] = set(query["entities"])
    return data


rel_linked_entities = read(rel_linked_entities_path)
dpr_nq_entities = read(dpr_nq_entities_path)
total_entities = 0
overlap_entities = 0

for qid in rel_linked_entities:
    total_entities += len(rel_linked_entities[qid])
    overlap_entities += len(
        rel_linked_entities[qid].intersection(dpr_nq_entities[qid]))
percentage = overlap_entities*1.0/total_entities
print(f"Overlap: {overlap_entities}/{total_entities} = {percentage*100}%")
