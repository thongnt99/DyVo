from pathlib import Path
import pandas as pd
import json
from tqdm import tqdm
import numpy as np


def filter_en_entities(emb_path="resources/wikipedia-20190701/enwiki-20190701-model-w2v-dim300"):
    name2text = {}
    with open("data/entity.jsonl", "r") as f:
        for line in tqdm(f):
            entity_obj = json.loads(line.strip())
            title = entity_obj["title"]
            entity_name = entity_obj["entity"]
            text = entity_obj["text"]
            if entity_name != title:
                print("---------")
                print(entity_name)
                print(title)
            name2text[entity_name] = title + " " + text
    entity_names = []
    entity_embs = []
    with open(emb_path, "r") as f:
        for line in tqdm(f):
            if line.startswith("ENTITY/"):
                tokens = line.strip().split(" ")
                ent_name = tokens[0][7:].replace("_", " ")
                if not ent_name in name2text:
                    continue
                ent_emb = [float(num) for num in tokens[1:]]
                entity_names.append(ent_name)
                entity_embs.append(ent_emb)
    entity_ids = list(range(len(entity_names)))
    df = pd.DataFrame(
        {"id": entity_ids, "entity": entity_names, "emb": entity_embs})
    print(f"{len(entity_names)}/{len(name2text)} entities have pretrained embeddings")
    cache_file = "data/enwiki-20190701-model-w2v-dim300.parquet"
    print(f"Saving embeddings to {str(cache_file)} in parquet format")
    df.to_parquet(cache_file, compression="gzip")
    with open("data/entity_anserini.jsonl", "w") as f:
        for ent_id, ent_name in zip(entity_ids, entity_names):
            f.write(json.dumps(
                {"id": ent_id, "text": name2text[ent_name]})+"\n")


if __name__ == "__main__":
    filter_en_entities()
