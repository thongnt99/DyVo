import pandas as pd
import ir_measures
from ir_measures import *
import ir_datasets
from collections import defaultdict
import json
path = "outputs/qmlp_dmlm_msmarco_hn_l1_0.0_0.0001/10-27-2023/model/msmarco/tmp_eval/eval_run.trec"
msmarco_run = list(ir_measures.read_trec_run(path))
run_dict = defaultdict(dict)
for row in msmarco_run:
    run_dict[row.query_id][row.doc_id] = row.score
dataset = ir_datasets.load("msmarco-passage/dev/small")
msmarco_qrels = list(dataset.qrels_iter())
eval_res = ir_measures.iter_calc([R@1000, MRR@10], msmarco_qrels, msmarco_run)
res = {}
zero_recalls = []
for item in eval_res:
    if item.measure == R@1000:
        res[item.query_id] = item.value
        if item.value == 0:
            zero_recalls.append(item.query_id)
json.dump(res, open("msmarco_r1000.json", "w"))

qdict = {}
for q in dataset.queries_iter():
    qdict[q.query_id] = q.text
q2rels = {}
for rel in msmarco_qrels:
    q2rels[rel.query_id] = rel.doc_id

zero_q_d = {}
queries = []
rel_docs = []
top_docs = []
with open("msmarco_q_d_zero_recalls.txt", "w") as f:
    for q_id in zero_recalls:
        rel_doc_id = q2rels[q_id]
        query = qdict[q_id]
        rel_doc = dataset.docs_store().get(rel_doc_id).text
        retrieved_docs = list(run_dict[q_id].keys())
        top_doc_id = sorted(
            retrieved_docs, key=lambda x: run_dict[q_id][x], reverse=True)[0]
        top_doc = dataset.docs_store().get(top_doc_id).text
        f.write("------------------------")
        f.write(f"Query ID: {q_id}\n")
        f.write(f"Query: {query}\n\n")
        f.write(f"Rel doc: {rel_doc}\n\n")
        f.write(f"Rank 1st: {top_doc}\n")
        queries.append(query)
        rel_docs.append(rel_doc)
        top_docs.append(top_doc)
df = pd.DataFrame({"qid": zero_recalls, "query": queries,
                  "rel doc": rel_docs, "top doc": top_docs})
df.to_csv("msmarco_zero_recalls.tsv", sep="\t")
