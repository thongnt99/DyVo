import json
import numpy as np
from tqdm import tqdm
from wikipedia2vec import Wikipedia2Vec
wiki2vec = Wikipedia2Vec.load('enwiki_20180420_100d.pkl')
entity_lists = json.load(open("data/entity_list.json"))
entity_embs = []
zeros = [0.0]*100
fails = []
for ent_name in tqdm(entity_lists):
    try:
        ent_vector = wiki2vec.get_entity_vector(ent_name).tolist()
    except:
        ent_vector = zeros
        fails.append(ent_name)
    entity_embs.append(ent_vector)
entity_embs = np.array(entity_embs)
print(f"number of failed entities: {len(fails)}")
json.dump(fails, open("failed_entities.json", "w"))
np.save("data/enwiki_20180420_100d_filtered.pkl",
        entity_embs, allow_pickle=True)
