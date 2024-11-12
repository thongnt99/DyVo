import sqlite3
db = sqlite3.connect("", isolation_level=None)
# from tqdm import tqdm
# import json
# from sqlitedict import SqliteDict
# pme_path = "data/wiki_p_m_e.json"
# db = SqliteDict(
#     "data/wiki_p_m_e1.sqlite", autocommit=False)
# pme_mapping = json.load(open(pme_path))
# count_m = 0
# for mention in tqdm(pme_mapping):
#     count_m += 1
#     db[mention] = pme_mapping[mention]
#     if count_m % 100000 == 0:
#         db.commit()
# db.commit()
# db.close()
