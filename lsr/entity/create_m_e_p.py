import json
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
aida_path = "rel/data/generic/p_e_m_data/aida_means.tsv"
crosswiki_path = "rel/data/generic/p_e_m_data/crosswikis_p_e_m.txt"
entity_id_path = "rel/data/wiki_2019/basic_data/wiki_name_id_map.txt"

id2entity = {}
ent_id_df = pd.read_csv(entity_id_path, delimiter="\t",
                        names=["entity", "id"], dtype=str)
id2entity = dict(zip(ent_id_df["id"], ent_id_df["entity"]))

m2e_mapping = defaultdict(lambda: defaultdict(lambda: 0))
aida_map = pd.read_csv(aida_path, delimiter="\t", names=[
                       "mention", "entity"], encoding="UTF-8", dtype=str)
for m, e in tqdm(zip(aida_map["mention"], aida_map["entity"])):
    if isinstance(m, str):
        e = e.replace("_", " ")
        m = m.lower()
        m2e_mapping[m][e] += 1
with open(crosswiki_path, "r") as f:
    for line in tqdm(f):
        tabs = line.strip().split("\t")
        m = tabs[0]
        total_freq = int(tabs[1])
        if total_freq <= 5:
            continue
        m = m.lower()
        for tab in tabs[2:]:
            entity_id, freq = tab.split(",")
            freq = int(freq)
            if entity_id in id2entity:
                entity = id2entity[entity_id]
                m2e_mapping[m][entity] += freq
entity_names = json.load(open("data/entity_list.json"))
entity_ids = list(range(len(entity_names)))
name2id = dict(zip(entity_names, entity_ids))
id2entity = dict(zip(entity_ids, entity_names))
res = {}
for m in tqdm(m2e_mapping):
    ent_cands = []
    total_count = 0
    for e, p in m2e_mapping[m].items():
        if e in name2id:
            ent_cands.append([name2id[e], p])
            total_count += p
    ent_cands = sorted(ent_cands, key=lambda x: x[1], reverse=True)[:200]
    for i in range(len(ent_cands)):
        ent_cands[i][1] /= total_count
    if len(ent_cands) > 0:
        res[m] = ent_cands
# for e, p in res["United States"][:5]:
#     print(id2entity[e], p)
for e, p in res["united states"][:5]:
    print(id2entity[e], p)
for e, p in res["us"][:5]:
    print(id2entity[e], p)
json.dump(res, open("data/wiki_p_m_e.json", "w"))
