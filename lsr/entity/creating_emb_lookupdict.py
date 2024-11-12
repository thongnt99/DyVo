
from tqdm import tqdm
from lsr.utils.ldict import PayloadLookup, encode_lz4json, decode_lz4json, encode_json, decode_json, encode_pickle, decode_pickle
import pandas as pd

emb_path = "data/enwiki-20190701-model-w2v-dim300.parquet"
emb_df = pd.read_parquet(emb_path)["emb"].tolist()


def iters():
    for i, emb in enumerate(emb_df):
        yield i, emb.tolist()


lookup = PayloadLookup.build(
    "data/wik_emb_json", iters(), encode_json, decode_json)
print(lookup['0'])
