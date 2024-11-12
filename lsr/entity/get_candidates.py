from tqdm import tqdm
from .candidate_filtering import CandidateRetriever
import json
import argparse
parser = argparse.ArgumentParser(description="Parsing arguments")
parser.add_argument("--inp", type=str, help="input path, or dataset")
parser.add_argument("--out", type=str, help="output path")
args = parser.parse_args()

cand_retriever = CandidateRetriever()

with open(args.inp, "r") as fin, open(args.out, "w") as fout:
    for line in tqdm(fin):
        text_id, text = line.strip().split("\t")
        entity_ids, entity_scores = cand_retriever.retrieve_candidates(text)
        out_obj = {"id": text_id, text: text,
                   "entity_ids": entity_ids, "entity_scores": entity_scores}
        fout.write(json.dumps(out_obj)+"\n")

# dataset = ir_datasets.load("msmarco-passage/train")
# with open("data/msmarco_train_queries.json", "w") as f:
#     for query in tqdm(list(dataset.queries_iter())):
#         q_id, q_text = query.query_id, query.text
#         cands = cand_retriever.retrieve_candidates(q_text)
#         json_res = {"id": q_id, "candidates": cands}
#         f.write(json.dumps(json_res)+"\n")
# with open("data/msmarco_passages.json", "w") as f:
#     for doc in tqdm(list(dataset.docs_iter())):
#         d_id, d_text = doc.doc_id, doc.text
#         cands = cand_retriever.retrieve_candidates(d_text)
#         json_res = {"id": q_id, "candidates": cands}
#         f.write(json.dumps(json_res)+"\n")

# dataset = ir_datasets.load("msmarco-passage/dev/small")
# with open("data/msmarco_dev_queries.json", "w") as f:
#     for query in tqdm(list(dataset.queries_iter())):
#         q_id, q_text = query.query_id, query.text
#         cands = cand_retriever.retrieve_candidates(q_text)
#         json_res = {"id": q_id, "candidates": cands}
#         f.write(json.dumps(json_res)+"\n")

# with open("data/candidates_msmarco-passage_passages.json", "w") as f:
#     for doc in dataset.docs_iter():
#         doc_id, d_text = doc.doc_id, doc.text
#         list_mentions = mention_detection.find_mentions(d_text, tagger_ngram)
#         entity2score = {}
#         for mention in list_mentions:
#             for ent_cand, ent_score in mention["candidates"]:
#                 if ent_cand in entity2score:
#                     entity2score[ent_cand] = max(
#                         entity2score[ent_cand], ent_score)
#                 else:
#                     entity2score[ent_cand] = ent_score
#         msmarco_entities.update(set(entity2score.keys()))
#         json_res = {"id": doc_id, "entities": entity2score}
#         f.write(json.dumps(json_res)+"\n")

# dataset = ir_datasets.load("msmarco-passage/dev/small")
# with open("data/candidates_msmarco-passage_dev_queries.json", "w") as f:
#     for query in dataset.queries_iter():
#         q_id, q_text = query.query_id, query.text
#         list_mentions = mention_detection.find_mentions(q_text, tagger_ngram)
#         entity2score = {}
#         for mention in list_mentions:
#             for ent_cand, ent_score in mention["candidates"]:
#                 if ent_cand in entity2score:
#                     entity2score[ent_cand] = max(
#                         entity2score[ent_cand], ent_score)
#                 else:
#                     entity2score[ent_cand] = ent_score
#         msmarco_entities.update(set(entity2score.keys()))
#         json_res = {"id": q_id, "entities": entity2score}
#         f.write(json.dumps(json_res)+"\n")

# json.dump(list(msmarco_entities), open("data/msmarco_entities.json", "w"))
