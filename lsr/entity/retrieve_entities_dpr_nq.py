import argparse
import torch
from datasets import load_dataset
from datasets import Features, Value, Sequence
import time
import faiss
import json
from glob import glob
# faiss.omp_set_num_threads(1)
parser = argparse.ArgumentParser(description="LSR Index Pisa")
parser.add_argument("--doc", type=str,
                    help="Document collection", default="data/entity_embs/*.parquet")
parser.add_argument("--query", type=str,
                    default="data/robust04/queries_dense_nq.parquet", help="Query path")
parser.add_argument("--output_path", type=str,
                    default="data/robust04/queries_ent_cand_dpr.jsonl", help="output path")
args = parser.parse_args()

# index = faiss.IndexHNSWFlat(256, 32, 0)
index = faiss.IndexFlatIP(768)

docs_data = load_dataset("parquet", data_files={
    "entities": args.doc}, keep_in_memory=True).with_format("numpy")
print(docs_data["entities"].features)
features = Features({'id': Value(dtype='string', id=None), 'emb': Sequence(
    feature=Value(dtype='float64', id=None), length=-1, id=None)})
queries_data = load_dataset("parquet", data_files={
                            "queries": args.query}, keep_in_memory=True, features=features).with_format("numpy")
index.add(docs_data["entities"]["emb"])
queries = queries_data["queries"]["emb"]

start = time.time()
D, I = index.search(queries, 1000)
end = time.time()
total_time = end - start
query_ids = queries_data["queries"]["id"].tolist()
ent_ids = docs_data["entities"]["id"].tolist()
with open(args.output_path, "w") as fout:
    for i in range(len(query_ids)):
        data = {"id": query_ids[i], "entities": []}
        for j in I[i]:
            data["entities"].append(ent_ids[j])
        fout.write(json.dumps(data)+"\n")
