from tqdm import tqdm
import json
query_path = "/projects/0/guse0488/dataset/entities/codec/query_entity.json"
doc_path = "/projects/0/guse0488/dataset/entities/codec/codec_entity_links.jsonl"
query_path_output = "/projects/0/guse0488/dataset/entities/codec/queries.jsonl"
doc_path_output = "/projects/0/guse0488/dataset/entities/codec/docs.jsonl"

# reading entities collection (from myself)
entities = json.load(open("data/entity_list.json"))
entity2id = dict(zip(entities, list(range(len(entities)))))


print("Procesing queries")
queries = json.load(open(query_path))
with open(query_path_output, "w") as fOut:
    for q_id in tqdm(queries):
        entity_names = queries[q_id]
        entity_ids = [entity2id[ent_name]
                      for ent_name in entity_names if ent_name in entity2id]
        q = {"id": q_id, "entities": list(set(entity_ids))}
        fOut.write(json.dumps(q)+"\n")

with open(doc_path, "r") as fIn, open(doc_path_output, "w") as fOut:
    for line in tqdm(fIn):
        doc_meta = json.loads(line.strip())
        doc_id = list(doc_meta.keys())[0]
        entity_list = set()
        for entity in doc_meta[doc_id]:
            predicted_entity = entity["prediction"].replace("_", " ")
            if predicted_entity in entity2id:
                entity_list.add(entity2id[predicted_entity])
        doc = {"id": doc_id, "entities": list(entity_list)}
        fOut.write(json.dumps(doc)+"\n")
