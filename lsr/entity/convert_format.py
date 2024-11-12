import json
from tqdm import tqdm
inp_path = "/projects/0/guse0488/dataset/mmead/msmarco_v1_passage_links_v1.0.json"
out_path = "/projects/0/guse0488/dataset/entities/msmarco-passage/msmarco_passages.jsonl"
entity_path = "data/entity_list.json"

entity_dict = json.load(open(entity_path, "r"))
entity_dict = dict(zip(entity_dict, range(len(entity_dict))))

# with open(inp_path, "r") as fin, open(out_path, "w") as fout:
#     for line in tqdm(fin):
#         doc = json.loads(line)
#         doc_id = doc["pid"]
#         entities = set()
#         for ent in doc["passage"]:
#             ent_name = ent["entity"]
#             if ent_name in entity_dict:
#                 ent_id = entity_dict[ent_name]
#                 entities.add(ent_id)
#         entities = list(entities)
#         doc = {"id": doc_id, "entities": entities}
#         fout.write(json.dumps(doc)+"\n")

train_query_path = "/projects/0/guse0488/dataset/mmead/topics.msmarco-passage.train.linked.json"
dev_query_path = "/projects/0/guse0488/dataset/mmead/topics.msmarco-passage.dev-subset.linked.json"
out_path = "/projects/0/guse0488/dataset/entities/msmarco-passage/msmarco_queries.jsonl"
with open(out_path, "w") as fout:
    with open(train_query_path, "r") as fin:
        for line in tqdm(fin):
            query = json.loads(line)
            query_id = query["qid"]
            entities = set()
            for ent in query["links"]:
                ent_name = ent["entity"]
                if ent_name in entity_dict:
                    ent_id = entity_dict[ent_name]
                    entities.add(ent_id)
            entities = list(entities)
            query = {"id": query_id, "entities": entities}
            fout.write(json.dumps(query)+"\n")
    with open(dev_query_path, "r") as fin:
        for line in tqdm(fin):
            query = json.loads(line)
            query_id = query["qid"]
            entities = set()
            for ent in query["links"]:
                ent_name = ent["entity"]
                if ent_name in entity_dict:
                    ent_id = entity_dict[ent_name]
                    entities.add(ent_id)
            entities = list(entities)
            query = {"id": query_id, "entities": entities}
            fout.write(json.dumps(query)+"\n")
