import argparse
import torch
import faiss
import json
from tqdm import tqdm
from datasets import load_dataset
from datasets import Features, Value, Sequence
import time

parser = argparse.ArgumentParser(description="LSR Index")
parser.add_argument("--query", type=str,
                    default="data/robust04/queries_blink.parquet", help="Query path")
parser.add_argument("--output_path", type=str,
                    default="data/robust04/queries_ent_cand_blink.jsonl", help="output path")
args = parser.parse_args()

features = Features({'id': Value(dtype='string', id=None), 'emb': Sequence(
    feature=Value(dtype='float64', id=None), length=-1, id=None)})
queries_data = load_dataset("parquet", data_files={
                            "queries": args.query}, keep_in_memory=True, features=features).with_format("numpy")
queries = queries_data["queries"]["emb"]

entity_collection_path = "/projects/0/guse0488/lsr-entities-project/BLINK/models/entity.jsonl"
index_path = "/projects/0/guse0488/lsr-entities-project/BLINK_index/faiss_flat_index.pkl"
print("Loading entity flat index")
entity_index = faiss.read_index(index_path)


def read_entity(ent_path):
    tmp_id2entity = {}
    tmp_id2text = {}
    with open(ent_path, "r") as f:
        for idx, line in enumerate(tqdm(f, desc="reading entity collection")):
            entity = json.loads(line)
            tmp_id2entity[idx] = entity["entity"]
            tmp_id2text[idx] = entity["text"]
    return tmp_id2entity, tmp_id2text


# blink entities
id2entity, id2text = read_entity(entity_collection_path)

# lsr entities: my own entity repository
lsr_entity = json.load(open("data/entity_list.json"))
lsr_ent2id = {ent: idx for idx, ent in enumerate(lsr_entity)}
start = time.time()
D, I = entity_index.search(queries, 1000)
end = time.time()
total_time = end - start
query_ids = queries_data["queries"]["id"].tolist()
# ent_ids = docs_data["entities"]["id"].tolist()
with open(args.output_path, "w") as fout:
    for i in range(len(query_ids)):
        data = {"id": query_ids[i], "entities": []}
        for j in I[i]:
            blink_entity = id2entity[j]
            if blink_entity in lsr_ent2id:
                lsr_ent_id = lsr_ent2id[blink_entity]
                data["entities"].append(lsr_ent_id)
        fout.write(json.dumps(data)+"\n")
