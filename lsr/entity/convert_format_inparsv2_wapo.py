import tqdm
import json
output_path = "/projects/0/guse0488/dataset/entities/wapo/queries_inparsv2_flair_ner.jsonl"
input_path = "data/wapo/inparsv2/queries-entities-flair-ner.json"
entity_list = json.load(open("data/entity_list.json", "r"))
entity2id = {ent_name: idx for idx, ent_name in enumerate(entity_list)}
rel_entities = json.load(open(input_path, "r"))
with open(output_path, "w") as f:
    for qid in rel_entities:
        entity_names = [entry[3] for entry in rel_entities[qid]]
        entity_ids = [entity2id[ent_name.replace(
            "_", " ")] for ent_name in entity_names if ent_name.replace(
            "_", " ") in entity2id]
        entity_ids = list(set(entity_ids))
        qentities = {"id": qid, "entities": entity_ids}
        f.write(json.dumps(qentities)+"\n")
