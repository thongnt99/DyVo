from pathlib import Path
from tqdm import tqdm
from ir_measures import *
import ir_measures
import ir_datasets
from collections import defaultdict
from collections import Counter
import json
query_cocondenser_path = "./outputs/qmlp_dmlm_robust04_cocondenser_msmarco_pretrained_inparsv2_monot53b_distillation_l1_0.0_0.00001/12-18-2023/tmp_eval/queries.tsv"
doc_cocondenser_path = "./outputs/qmlp_dmlm_robust04_cocondenser_msmarco_pretrained_inparsv2_monot53b_distillation_l1_0.0_0.00001/12-18-2023/tmp_eval/docs.jsonl"
query_ent_path = "./outputs/qmlp_dmlm_emlm_only_ent_dim100_robust04_msmarco_pretrained_inparsv2_monot53b_distillation_l1_0.0_0.00001/12-29-2023/tmp_eval/queries.jsonl"
doc_ent_path = "./outputs/qmlp_dmlm_emlm_only_ent_dim100_robust04_msmarco_pretrained_inparsv2_monot53b_distillation_l1_0.0_0.00001/12-29-2023/tmp_eval/docs.jsonl"
queries = {}
docs = {}
with open(query_cocondenser_path, "r") as f:
    for line in f:
        q_id, qtoks = line.strip().split("\t")
        q_toks = qtoks.split(" ")
        queries[q_id] = Counter(q_toks)


def read_jsonl(fp, scale=1.0):
    res = {}
    with open(fp, "r") as f:
        for line in tqdm(f, desc=f"Reading {fp}"):
            jsonrow = json.loads(line.strip())
            vector = {k: v * scale for k, v in jsonrow["vector"].items()}
            res[jsonrow["id"]] = vector
    return res


def read_and_score(fp, queries_to_score, scale=1.0, score_dict=None, cache_file="query2doc_score_word.json"):
    cache_file = Path(cache_file)
    if cache_file.exists():
        print(f"Reading scores from cache: {cache_file}")
        return json.load(open(cache_file))
    q2dscores = defaultdict(lambda: defaultdict(lambda: 0))
    with open(fp, "r") as f:
        for line in tqdm(list(f), desc=f"Reading and scoring {fp}"):
            jsonrow = json.loads(line.strip())
            doc_vector = jsonrow["vector"]
            doc_id_to_score = jsonrow["id"]
            for q_id in queries_to_score:
                score = 0
                for word in queries_to_score[q_id]:
                    if word in doc_vector:
                        if not score_dict:
                            score += queries_to_score[q_id][word] * \
                                doc_vector[word]
                        else:
                            if doc_id_to_score in score_dict[q_id]:
                                score += score_dict[q_id][doc_id_to_score]*queries_to_score[q_id][word] * \
                                    doc_vector[word]
                            # break
                score = score * scale
                if score > 0:
                    q2dscores[q_id][doc_id_to_score] = score
    json.dump(q2dscores, open(cache_file, "w"))
    return q2dscores


q2dscores_word = read_and_score(
    doc_cocondenser_path, queries, cache_file="q2d_word.json")
query_entities = read_jsonl(query_ent_path, scale=1.0)
q2dscores_ent = read_and_score(
    doc_ent_path, query_entities, scale=1000.0, score_dict=q2dscores_word, cache_file="q2d_entity.json")
# for qid in q2dscores_word:
#     pairs = sorted(q2dscores_word[qid].items(),
#                    key=lambda x: x[1], reverse=True)[:20000]
#     q2dscores_word[qid] = dict(pairs)

qrels = ir_datasets.load("disks45/nocr/trec-robust-2004").qrels_iter()
metrics = ir_measures.calc_aggregate(
    [NDCG@10, NDCG@20, R@1000, R@100], qrels, q2dscores_ent)
print(metrics)
# for idx in range(0, 11, 2):
#     alpha = idx*0.1
#     print(f"Alpha: {alpha}:")
#     run = {}
#     for qid in q2dscores_word:
#         run[qid] = Counter(
#             {k: v*alpha for k, v in q2dscores_word[qid].items() if (v*alpha) > 0})
#         run[qid].update({k: v*1000*(1-alpha)
#                         for k, v in q2dscores_ent[qid].items() if v*1000*(1-alpha) > 0})
#     qrels = ir_datasets.load("disks45/nocr/trec-robust-2004").qrels_iter()
#     metrics = ir_measures.calc_aggregate(
#         [NDCG@10, NDCG@20, R@1000, R@100], qrels, run)
#     print(metrics)
