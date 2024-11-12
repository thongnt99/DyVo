import json
from tqdm import tqdm

title_query_path = "/projects/0/guse0488/dataset/entities/trec-robust04/title.query_annotations.tsv"
desc_query_path = "/projects/0/guse0488/dataset/entities/trec-robust04/desc.query_annotations.tsv"
doc_path = "/projects/0/guse0488/dataset/entities/trec-robust04/robust04.entity_links.jsonl"


title_output_path = "/projects/0/guse0488/dataset/entities/trec-robust04/queries_title.jsonl"
desc_output_path = "/projects/0/guse0488/dataset/entities/trec-robust04/queries_desc.jsonl"
doc_output_path = "/projects/0/guse0488/dataset/entities/trec-robust04/docs.jsonl"

# my own entity repository (subset of wikipedia)
entity_dict = json.load(open("data/entity_list.json", "r"))
entity2id = dict(zip(entity_dict, list(range(len(entity_dict)))))

print("Processing queries' title")
with open(title_query_path, "r") as fIn, open(title_output_path, "w") as fOut:
    for line in tqdm(fIn):
        qid, entity_json = line.strip().split("\t")
        entity_list = []
        for entity in json.loads(entity_json):
            if entity["entity_name"] in entity2id:
                entity_list.append(entity2id[entity["entity_name"]])
        qent = {"id": qid, "entities": entity_list}
        fOut.write(json.dumps(qent)+"\n")
print("Processing queries' desc")
with open(desc_query_path, "r") as fIn, open(desc_output_path, "w") as fOut:
    for line in tqdm(fIn):
        qid, entity_json = line.strip().split("\t")
        entity_list = []
        for entity in json.loads(entity_json):
            if entity["entity_name"] in entity2id:
                entity_list.append(entity2id[entity["entity_name"]])
        qent = {"id": qid, "entities": entity_list}
        fOut.write(json.dumps(qent)+"\n")


# reading meta data from Shubham
print("Loading meta data")
entity_meta_path = "/projects/0/guse0488/dataset/entities/entities.jsonl"
pageid2entity = {}
with open(entity_meta_path, "r") as f:
    for line in tqdm(f):
        wiki = json.loads(line)
        page_id = str(wiki["page_id"])
        title = wiki["title"]
        if title in entity2id:
            pageid2entity[page_id] = title
        else:
            title = wiki["redirect_title"]
            if title != "None" and title in entity2id:
                pageid2entity[page_id] = title

print("Processing documents")
with open(doc_path, "r") as fIn, open(doc_output_path, "w") as fOut:
    for line in tqdm(fIn):
        doc = json.loads(line.strip())
        entity_list = []
        for page_id in doc["entities"]:
            if page_id in pageid2entity:
                entity_name = pageid2entity[page_id]
                if entity_name in entity2id:
                    entity_list.append(entity2id[entity_name])
        doc = {"id": doc["doc_id"], "entities": entity_list}
        fOut.write(json.dumps(doc)+"\n")
