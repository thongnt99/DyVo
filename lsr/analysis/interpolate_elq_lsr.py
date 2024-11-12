import argparse
import ir_datasets
import ir_measures
from ir_measures import *
import json
from tqdm import tqdm
from glob import glob
from pathlib import Path
parser = argparse.ArgumentParser("Parsing arguments")
parser.add_argument("--wq", type=str, default="/scratch-shared/tnguyen/lsr-entities/outputs/qmlp_dmlm_wapo_msmarco_pretrained_inparsv2_monot53b_distillation_l1_0.0_0.00001/eval/tmp_eval/queries.jsonl")
parser.add_argument(
    "--wd", type=str, default="/scratch-shared/tnguyen/lsr-entities/outputs/qmlp_dmlm_wapo_msmarco_pretrained_inparsv2_monot53b_distillation_l1_0.0_0.00001/eval/tmp_eval/docs/*")
parser.add_argument("--eq", type=str, default="/scratch-shared/tnguyen/lsr-entities/outputs/qmlp_dmlm_emlm.elq_wapo_msmarco_pretrained_inparsv2_monot53b_distillation_l1_noent_0.0_0.001/eval/tmp_eval/queries.jsonl")
parser.add_argument("--ed", type=str, default="/scratch-shared/tnguyen/lsr-entities/outputs/qmlp_dmlm_emlm.elq_wapo_msmarco_pretrained_inparsv2_monot53b_distillation_l1_noent_0.0_0.001/eval/tmp_eval/docs/*")
parser.add_argument("--qrels", type=str, default="wapo/v2/trec-core-2018")
args = parser.parse_args()


def read_jsonl(file_path, word=True):
    id2vec = {}
    with open(file_path, "r") as fin:
        for line in fin:
            data = json.loads(line.strip())
            id2vec[data["id"]] = {
                k: v for k, v in data["vector"].items() if word or int(k) >= 30522}
    return id2vec


word_qid2vec = read_jsonl(args.wq)
word_did2vec = {}
for doc_path in tqdm(glob(args.wd), desc="reading word-level document representation"):
    part = read_jsonl(doc_path)
    word_did2vec.update(part)

ent_qid2vec = read_jsonl(args.eq, word=False)
ent_did2vec = {}
for doc_path in tqdm(glob(args.ed), desc="reading ent-level document representation"):
    part = read_jsonl(doc_path, word=False)
    ent_did2vec.update(part)

word_scores = {}
for qid in word_qid2vec:
    word_scores[qid] = {}
    for did in word_did2vec:
        score = 0
        for tok in word_qid2vec[qid]:
            if tok in word_did2vec[did]:
                score += word_qid2vec[qid][tok] * word_did2vec[did][tok]
        word_scores[qid][did] = score

ent_scores = {}
for qid in ent_qid2vec:
    ent_scores[qid] = {}
    for did in ent_did2vec:
        score = 0
        for tok in ent_qid2vec[qid]:
            if tok in ent_did2vec[did]:
                score += ent_qid2vec[qid][tok] * ent_did2vec[did][tok]
        ent_scores[qid][did] = score

qrels = list(ir_datasets.load(args.qrels).qrels_iter())
entity_metrics = ir_measures.calc_aggregate(
    [NDCG@10, NDCG@20, R@100, R@1000], qrels, ent_scores)
print("Entiy scores:", entity_metrics)

word_metrics = ir_measures.calc_aggregate(
    [NDCG@10, NDCG@20, R@100, R@1000], qrels, word_scores)
print("Entiy scores:", word_metrics)

for v in range(1, 10):
    alpha = v * 0.1
    fusion_scores = {}
    for qid in word_scores:
        fusion_scores[qid] = {}
        for did in word_scores[qid]:
            fusion_scores[qid][did] = word_scores[qid][did]
            if did in ent_scores[qid]:
                fusion_scores[qid][did] += alpha * ent_scores[qid][did]
        for did in ent_scores[qid]:
            if not did in word_scores[qid]:
                fusion_scores[qid][did] = alpha * ent_scores[qid][did]
    fusion_metrics = ir_measures.calc_aggregate(
        [NDCG@10, NDCG@20, R@100, R@1000], qrels, fusion_scores)
    print(f"Fusion scores alpha = {alpha}:", fusion_metrics)
