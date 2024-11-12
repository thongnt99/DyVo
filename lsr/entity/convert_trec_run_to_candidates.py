import ir_measures
from collections import defaultdict
import json
from tqdm import tqdm
inp_query = "entity_retrieval/bm25/msmarco_queries.trec"
out_query = "/projects/0/guse0488/dataset/entities/msmarco-passage/msmarco_queries_bm25.jsonl"
inp_passage = "entity_retrieval/bm25/msmarco_passages.trec"
out_passage = "/projects/0/guse0488/dataset/entities/msmarco-passage/msmarco_passages_bm25.jsonl"


def convert(inp_path, out_path):
    id2candidates = defaultdict(list)
    id2scores = defaultdict(list)
    run = ir_measures.read_trec_run(inp_path)
    for row in tqdm(run):
        id2candidates[row.query_id].append(row.doc_id)
        id2scores[row.query_id].append(row.score)
    with open(out_path, "w") as f:
        for query_id in tqdm(id2candidates):
            obj = {"id": query_id,
                   "entities": id2candidates[query_id], "scores": id2scores[query_id]}
            f.write(json.dumps(obj)+"\n")


convert(inp_query, out_query)
convert(inp_passage, out_passage)
