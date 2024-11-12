import json
from tqdm import tqdm
from sqlitedict import SqliteDict
db = SqliteDict(
    "resources/wikipedia-20190701/enwiki-20190701-model-w2v-dim300-ent.sqlite", outer_stack=False)
id2entity = {}
c_entity = 0
c_overlap = 0
with open("rel/data/wiki_2019/basic_data/wiki_name_id_map.txt", "r") as f:
    for line in tqdm(f):
        c_entity += 1
        entity_name, entity_id = line.strip().split("\t")
        if entity_name in db:
            c_overlap += 1
            id2entity[entity_id] = entity_name
print(f"{c_overlap}/{c_entity} common entities")
json.dump(id2entity, open("rel/data/wiki_2019/basic_data/id2entity.json", "w"))
