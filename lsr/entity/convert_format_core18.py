from tqdm import tqdm
import json
query_path = "/projects/0/guse0488/dataset/entities/trec-core18/query_annotations.tsv"
doc_path = "/projects/0/guse0488/dataset/entities/trec-core18/corpus.entity_links.jsonl"
query_path_output = "/projects/0/guse0488/dataset/entities/trec-core18/queries.jsonl"
doc_path_output = "/projects/0/guse0488/dataset/entities/trec-core18/docs.jsonl"

# reading entities collection (from myself)
entities = json.load(open("data/entity_list.json"))
entity2id = dict(zip(entities, list(range(len(entities)))))


# my own entity repository (subset of wikipedia)
entity_dict = json.load(open("data/entity_list.json", "r"))
entity2id = dict(zip(entity_dict, list(range(len(entity_dict)))))

print("Processing queries")
with open(query_path, "r") as fIn, open(query_path_output, "w") as fOut:
    for line in tqdm(fIn):
        qid, entity_json = line.strip().split("\t")
        entity_list = set()
        for entity in json.loads(entity_json):
            if entity["entity_name"] in entity2id:
                entity_list.add(entity2id[entity["entity_name"]])
        qent = {"id": qid, "entities": list(entity_list)}
        fOut.write(json.dumps(qent)+"\n")

with open(doc_path, "r") as fIn, open(doc_path_output, "w") as fOut:
    for line in tqdm(fIn):
        doc_meta = json.loads(line.strip())
        doc_id = doc_meta["doc_id"]
        entity_list = set()
        for entity in doc_meta["entities"]:
            predicted_entity = entity["wikipedia_title"].replace("_", " ")
            if predicted_entity in entity2id:
                entity_list.add(entity2id[predicted_entity])
        doc = {"id": doc_id, "entities": list(entity_list)}
        fOut.write(json.dumps(doc)+"\n")
