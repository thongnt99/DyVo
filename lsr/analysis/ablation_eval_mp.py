import argparse
import json
from collections import defaultdict
from glob import glob
from pathlib import Path
from tqdm import tqdm
from ir_measures import *
import ir_measures
from heapq import heappop, heappush
import ir_datasets
from heapq import heappush, heappop
from multiprocessing import Pool


def load_docs(df):
    with open(df, "r") as f:
        return [json.loads(line.strip()) for line in f]


def score_documents(q):
    qid, q_vector = q["id"], q["vector"]
    scores = []
    for d in docs:
        score = 0.0
        for tok in q_vector:
            if int(tok) < 30522 and tok in d["vector"]:
                score += q_vector[tok] * d["vector"][tok]
        if len(scores) < 1000:
            heappush(scores, (score, d["id"]))
        else:
            if score > scores[0][0]:
                heappop(scores)
                heappush(scores, (score, d["id"]))
    return qid, scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate results without entities")
    parser.add_argument("--q_path", type=str, required=True)
    parser.add_argument("--d_dir", type=str, required=True)
    parser.add_argument("--qrel", type=str, default="disks45/nocr/trec-robust-2004", required=False,
                        help="add irds:: prefix to load qrels from ir_datasets")
    args = parser.parse_args()
    docs = []
    for fn in tqdm(glob(str(Path(args.d_dir)/"*")), desc="Loading documents"):
        docs.extend(load_docs(fn))
    # Load queries
    with open(args.q_path, "r") as f:
        queries = [json.loads(line.strip()) for line in f]

    # Score documents in parallel using multiprocessing
    with Pool(16) as pool:
        scores = pool.map(score_documents, queries)

    run = {qid: {k: v for v, k in doc_scores} for qid, doc_scores in scores}

    qrels = list(ir_datasets.load(args.qrel).qrels_iter())
    metrics = ir_measures.calc_aggregate(
        [NDCG@10, NDCG@20, R@100, R@1000], qrels, run)
    print(metrics)
