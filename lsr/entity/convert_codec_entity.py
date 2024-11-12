
import tqdm
import ir_measures
import json
from collections import defaultdict
query_entity_path = "/projects/0/guse0488/dataset/entities/codec/entity_ndcg.qrels"
wiki_kb = "/projects/0/guse0488/dataset/entities/kilt/kilt_knowledgesource.json"

entity_dict = {}
with open(wiki_kb, "r") as f:
    for line in f:
        wiki_page = json.loads(line)
        entity_id = wiki_page["_id"]
        entity_name = wiki_page["wikipedia_title"]
        entity_dict[entity_id] = entity_name
qrels = ir_measures.read_trec_qrels(query_entity_path)
qid2entities = defaultdict(list)
for row in qrels:
    if row.relevance > 1:
        qid2entities[row.query_id].append(entity_dict[row.doc_id])

output_path = "/projects/0/guse0488/dataset/entities/codec/query_entity.json"
json.dump(qid2entities, open(output_path, "w"))
