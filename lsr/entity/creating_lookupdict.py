
from tqdm import tqdm
import json
from lsr.utils.ldict import PayloadLookup, encode_lz4json, decode_lz4json
from sqlitedict import SqliteDict
pme_path = "data/wiki_p_m_e.json"
pme_mapping = json.load(open(pme_path))
lookup = PayloadLookup.build(
    "data/wiki_pme", pme_mapping.items(), encode_lz4json, decode_lz4json)
print(lookup["united states"])
