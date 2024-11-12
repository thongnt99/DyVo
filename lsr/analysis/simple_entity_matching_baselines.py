import ir_datasets
from ir_measures import *
import ir_measures
import math
from pathlib import Path
from collections import defaultdict
import json
import subprocess


def read_entities(fn):
    id2entites = defaultdict(set)
    with open(fn, "r") as f:
        for line in f:
            row = json.loads(line.strip())
            if len(row["entities"]) > 0:
                if row["id"].startswith("inparsv2_"):
                    row['id'] = row["id"].replace("inparsv2_", "")
                id2entites[row["id"]] = set(row["entities"])
    return id2entites


corpus = Path("outputs/simple_baseline_w=1")
if not corpus.is_dir():
    corpus.mkdir(parents=True)
query_entities = read_entities(
    "/projects/0/guse0488/dataset/entities/trec-robust04/queries_desc.jsonl")
doc_entities = read_entities(
    "/projects/0/guse0488/dataset/entities/trec-robust04/docs.jsonl")
queries_path = corpus/"queries.tsv"
# with open(queries_path, "w") as f:
#     for qid in query_entities:
#         qtext = " ".join([str(eid) for eid in query_entities[qid]])
#         f.write(f"{qid}\t{qtext}\n")
doc_ids = list(doc_entities.keys())
partition_size = math.ceil(len(doc_ids)*1.0/10)
docs_corpus = corpus/"docs"
# if not docs_corpus.is_dir():
#     docs_corpus.mkdir()
# for i in range(10):
#     with open(docs_corpus/f"part{i}.jsonl", "w") as f:
#         for doc_id in doc_ids[i*partition_size: (i+1)*partition_size]:
#             if len(doc_entities[doc_id]) == 0:
#                 continue
#             row = {"id": doc_id, "vector": {
#                 str(ent_id): 1 for ent_id in doc_entities[doc_id]}}
#             row_text = json.dumps(row)
#             f.write(row_text+"\n")
index_dir = corpus/"idnex"
# ANSERINI_INDEX_COMMAND = f"""/projects/0/guse0488/anserini-lsr/target/appassembler/bin/IndexCollection
#         -collection JsonSparseVectorCollection
#         -input {docs_corpus}
#         -index {index_dir}
#         -generator SparseVectorDocumentGenerator
#         -threads 18
#         -impact
#         -pretokenized
#     """
# process = subprocess.run(ANSERINI_INDEX_COMMAND.split(), check=True)
run_path = corpus/"test_run.trec"
ANSERINI_RETRIEVE_COMMAND = f"""/projects/0/guse0488/anserini-lsr/target/appassembler/bin/SearchCollection
    -index {index_dir}  
    -topics {queries_path} 
    -topicreader TsvString 
    -output {run_path}  
    -impact 
    -pretokenized 
    -hits 100000
    -parallelism 18"""
process = subprocess.run(ANSERINI_RETRIEVE_COMMAND.split(), check=True)
run = list(ir_measures.read_trec_run(str(run_path)))
qrels = ir_datasets.load("disks45/nocr/trec-robust-2004").qrels_iter()
metrics = ir_measures.calc_aggregate(
    [MAP, MRR@10, R@5, R@10, R@100, R@1000, R@100000, NDCG@10, NDCG@20], qrels, run)
print(metrics)
