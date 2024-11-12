import argparse
import json
from glob import glob
from tqdm import tqdm
import ir_datasets
import ir_measures
from ir_measures import *

parser = argparse.ArgumentParser("")
parser.add_argument("--query", type=str, default="/scratch-shared/nthong/lsr-entities/outputs/qmlp_dmlm_emlm.cls_robust04_msmarco_pretrained_inparsv2_monot53b_distillation_l1_noent_0.0_0.00001/02-21-2024/tmp_eval/queries.jsonl")
parser.add_argument("--doc", type=str, default="/scratch-shared/nthong/lsr-entities/outputs/qmlp_dmlm_emlm.cls_robust04_msmarco_pretrained_inparsv2_monot53b_distillation_l1_noent_0.0_0.00001/02-21-2024/tmp_eval/docs/*")
parser.add_argument("--qrel", type=str,
                    default="disks45/nocr/trec-robust-2004")
args = parser.parse_args()


def read_jsonl(file_path):
    id2vec = {}
    with open(file_path, "r") as fin:
        for line in fin:
            data = json.loads(line.strip())
            id2vec[data["id"]] = data["vector"]
    return id2vec


qid2vec = read_jsonl(args.query)
did2vec = {}
for doc_path in tqdm(glob(args.doc), desc="reading document representation"):
    part = read_jsonl(doc_path)
    did2vec.update(part)

run = {}
for qid in tqdm(qid2vec, desc="Re-evaluating queries without entities"):
    run[qid] = {}
    for docid in did2vec:
        score = 0.0
        for tok_id in qid2vec[qid]:
            if int(tok_id) < 30522 and tok_id in did2vec[docid]:
                score += qid2vec[qid][tok_id] * did2vec[docid][tok_id]
        run[qid][docid] = score

qrels = ir_datasets.load_dataset(args.qrel).qrels_iter()

metrics = ir_measures.calc_aggregate(
    [NDCG@10, NDCG@20, R@100, R@1000], qrels, run)
print(metrics)
