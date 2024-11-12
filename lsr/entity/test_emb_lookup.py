import pandas as pd
import random
from lsr.utils.ldict import PayloadLookup, encode_lz4json, decode_lz4json, decode_json, decode_pickle
import time
random.seed(1000)

ids = []

for _ in range(1, 1000):
    idx = random.randint(0, 5000000)
    ids.append(idx)

# lookup = PayloadLookup("data/wik_emb", decode_lz4json)
# start = time.time()
# for idx in ids:
#     emb = lookup[str(idx)]
# end = time.time()
# print("PayloadLookup + lz4json:", end-start)

# inmem = pd.read_parquet(
#     "data/enwiki-20190701-model-w2v-dim300.parquet")["emb"].tolist()
# start = time.time()
# for idx in ids:
#     emb = inmem[idx]
# end = time.time()
# print("In-memory: ", end-start)

lookup = PayloadLookup("data/wik_emb_json", decode_json)
start = time.time()
for idx in ids:
    emb = lookup[str(idx)]
end = time.time()
print("PayloadLookup + json:", end-start)
