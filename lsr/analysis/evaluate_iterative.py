import argparse
import json
import ir_datasets
from collections import defaultdict
from glob import glob
from pathlib import Path
from tqdm import tqdm
from ir_measures import *
import ir_measures
from heapq import heappush, heappop
parser = argparse.ArgumentParser("Evaluate results without entities")
parser.add_argument("--q_path", type=str, required=True)
parser.add_argument("--d_dir", type=str, required=True)
parser.add_argument("--qrel", type=str, default="disks45/nocr/trec-robust-2004", required=False,
                    help="add irds:: prefix to load qrels from ir_datasets")
args = parser.parse_args()

docs = []
for df in tqdm(glob(str(Path(args.d_dir)/"*")), desc="Loading document representations"):
    with open(df, "r") as f:
        for line in f:
            docs.append(json.loads(line.strip()))
scores = defaultdict(list)
with open(args.q_path, "r") as f:
    for line in tqdm(f, desc="Scoring q-d pairs"):
        q = json.loads(line.strip())
        for d in docs:
            score = 0.0
            for tok in q["vector"]:
                if tok in d["vector"]:
                    score += q["vector"][tok] * d["vector"][tok]
            if len(scores[q["id"]]) < 1000:
                heappush(scores[q["id"]], (score, d["id"]))
            else:
                if score > scores[q["id"]][0][0]:
                    heappop(scores[q["id"]])
                    heappush(scores[q["id"]], (score, d["id"]))
run = {}
for qid in scores:
    run[qid] = {k: v for v, k in scores[qid]}
qrels = list(ir_datasets.load(args.qrel).qrels_iter())
metrics = ir_measures.calc_aggregate(
    [MRR@10, NDCG@10, NDCG@20, R@100, R@1000], qrels, run)
print(metrics)
