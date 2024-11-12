import argparse
import json
import ir_datasets
import ir_measures
from collections import defaultdict
from glob import glob
from pathlib import Path
from tqdm import tqdm
from ir_measures import *
import ir_measures
import subprocess
import tempfile
import os
from pathlib import Path
from heapq import heappush, heappop
parser = argparse.ArgumentParser("Evaluate results without entities")
parser.add_argument("--q_path", type=str, required=True)
parser.add_argument("--d_dir", type=str, required=True)
parser.add_argument("--qfactor", type=float, default=51.2)
parser.add_argument("--threads", type=int, default=18)
parser.add_argument("--qrel", type=str, default="msmarco-passage/dev/small", required=False,
                    help="add irds:: prefix to load qrels from ir_datasets")
args = parser.parse_args()


tmp_dir = Path("anserini_temp")
tmp_dir.mkdir()
query_path = tmp_dir/"queries.tsv"
docs_dir = tmp_dir/"docs"
index_dir = tmp_dir/"index"
run_path = tmp_dir/"run.trec"
docs_dir.mkdir()
index_dir.mkdir()

for df in tqdm(glob(str(Path(args.d_dir)/"*")), desc=f"Load docs with raw weights from: {args.d_dir} and write quantized weights to: {docs_dir}"):
    outf = docs_dir/os.path.basename(df)
    with open(df, "r") as fin, open(outf, "w") as fout:
        for line in fin:
            doc = json.loads(line.strip())
            quantized_vector = {k: int(v*args.qfactor)
                                for k, v in doc["vector"].items() if int}
            quantized_vector = {k: v for k,
                                v in quantized_vector.items() if v > 0}
            doc["vector"] = quantized_vector
            fout.write(json.dumps(doc)+"\n")
with open(args.q_path, "r") as fin, open(query_path, "w") as fout:
    for line in tqdm(fin, desc=f"Quantizing query weights and writing to: {query_path}"):
        query = json.loads(line.strip())
        qid = query["id"]
        qtext = []
        for tok, w in query["vector"].items():
            qtext.extend([tok] * int(w*args.qfactor))
        qtext = " ".join(qtext)
        fout.write(f"{qid}\t{qtext}\n")

ANSERINI_INDEX_COMMAND = f"""/projects/0/guse0488/anserini-lsr/target/appassembler/bin/IndexCollection
    -collection JsonSparseVectorCollection 
    -input {docs_dir}  
    -index {index_dir}  
    -generator SparseVectorDocumentGenerator 
    -threads {args.threads} 
    -impact 
    -pretokenized
"""
process = subprocess.run(ANSERINI_INDEX_COMMAND.split(), check=True)
ANSERINI_RETRIEVE_COMMAND = f"""/projects/0/guse0488/anserini-lsr/target/appassembler/bin/SearchCollection
    -index {index_dir}  
    -topics {query_path} 
    -topicreader TsvString 
    -output {run_path}  
    -impact 
    -pretokenized 
    -hits 1000 
    -parallelism {args.threads}"""

process = subprocess.run(ANSERINI_RETRIEVE_COMMAND.split(), check=True)
run = list(ir_measures.read_trec_run(str(run_path)))
qrels = list(ir_datasets.load(args.qrel).qrels_iter())
metrics = ir_measures.calc_aggregate(
    [MRR@10, NDCG@10, NDCG@20, R@100, R@1000], qrels, run)
print(metrics)
