from .candidate_filtering import CandidateRetriever
import json
from tqdm import tqdm
import ir_datasets
#############################################################################################
#                                      MSMARCO queries                                      #
#############################################################################################

# rel_ann_path = "/projects/0/guse0488/dataset/mmead/topics.msmarco-passage.dev-subset.linked.json"
# ground_truth = {}
# with open(rel_ann_path, "r") as f:
#     for line in f:
#         query = json.loads(line.strip())
#         ground_truth[query["qid"]] = set()
#         for ent in query["links"]:
#             ground_truth[query["qid"]].add(ent["entity"])

# entity_collection = set()
# num_candidates = []
# recalls = []
# cand_retriever = CandidateRetriever()
# entitydict = json.load(open("data/entity_list.json"))
# dataset = ir_datasets.load("msmarco-passage/dev/small")
# for query in tqdm(list(dataset.queries_iter())):
#     q_id, q_text = query.query_id, query.text
#     entity_ids, *_ = cand_retriever.retrieve_candidates(q_text)
#     entity_names = [entitydict[eid] for eid in entity_ids]
#     num_candidates.append(len(entity_ids))
#     entity_collection.update(entity_names)
#     if q_id in ground_truth and len(ground_truth[q_id]) > 0:
#         recall = len(set(entity_names).intersection(
#             ground_truth[q_id]))/len(ground_truth[q_id])
#         recalls.append(recall)


# avg_num_cands = sum(num_candidates)/len(num_candidates)
# avg_recall = sum(recalls)/len(recalls)
# total_entities = len(entity_collection)
# print(f"Number of candidates per query: {avg_num_cands}")
# print(f"Recall per query: {avg_recall}")
# print(f"Total number of entities: {total_entities}")
# json.dump(list(entity_collection), open("data/msmarco_entities.json", "w"))


#############################################################################################
#                                      MSMARCO passages                                     #
#############################################################################################

# rel_ann_path = "/projects/0/guse0488/dataset/mmead/msmarco_v1_passage_links_v1.0.json"
# ground_truth = {}
# with open(rel_ann_path, "r") as f:
#     for line in f:
#         passage = json.loads(line.strip())
#         ground_truth[str(passage["pid"])] = set()
#         for ent in passage["passage"]:
#             ground_truth[str(passage["pid"])].add(ent["entity"])

# entity_collection = set()
# num_candidates = []
# recalls = []
# cand_retriever = CandidateRetriever()
# entitydict = json.load(open("data/entity_list.json"))
# dataset = ir_datasets.load("msmarco-passage/dev/small")
# for query in tqdm(list(dataset.docs_iter())):
#     p_id, p_text = query.doc_id, query.text
#     entity_ids, *_ = cand_retriever.retrieve_candidates(p_text)
#     entity_names = [entitydict[eid] for eid in entity_ids]
#     num_candidates.append(len(entity_ids))
#     entity_collection.update(entity_names)
#     if p_id in ground_truth and len(ground_truth[p_id]) > 0:
#         recall = len(set(entity_names).intersection(
#             ground_truth[p_id]))/len(ground_truth[p_id])
#         recalls.append(recall)
#     if len(recalls) > 20000:
#         break

# avg_num_cands = sum(num_candidates)/len(num_candidates)
# avg_recall = sum(recalls)/len(recalls)
# total_entities = len(entity_collection)
# print(f"Number of candidates per query: {avg_num_cands}")
# print(f"Recall per query: {avg_recall}")
# print(f"Total number of entities: {total_entities}")
# json.dump(list(entity_collection), open("data/msmarco_entities.json", "w"))


#############################################################################################
# #                                      Robust04 title                                       #
#############################################################################################

# rel_ann_path = "/projects/0/guse0488/dataset/entities/trec-robust04/title.query_annotations.tsv"
# ground_truth = {}
# with open(rel_ann_path, "r") as f:
#     for line in f:
#         qid, entities = line.strip().split("\t")
#         entities = json.loads(entities)
#         ground_truth[qid] = set()
#         for ent in entities:
#             ground_truth[qid].add(ent["entity_name"])

# entity_collection = set()
# num_candidates = []
# recalls = []
# cand_retriever = CandidateRetriever()
# entitydict = json.load(open("data/entity_list.json"))
# dataset = ir_datasets.load("disks45/nocr/trec-robust-2004")
# for query in tqdm(list(dataset.queries_iter())):
#     q_id, q_text = query.query_id, query.title
#     entity_ids, *_ = cand_retriever.retrieve_candidates(q_text)
#     entity_names = [entitydict[eid] for eid in entity_ids]
#     num_candidates.append(len(entity_ids))
#     entity_collection.update(entity_names)
#     if q_id in ground_truth and len(ground_truth[q_id]) > 0:
#         recall = len(set(entity_names).intersection(
#             ground_truth[q_id]))/len(ground_truth[q_id])
#         recalls.append(recall)
#     if len(recalls) > 20000:
#         break

# avg_num_cands = sum(num_candidates)/len(num_candidates)
# avg_recall = sum(recalls)/len(recalls)
# total_entities = len(entity_collection)
# print(f"Number of candidates per query: {avg_num_cands}")
# print(f"Recall per query: {avg_recall}")
# print(f"Total number of entities: {total_entities}")


#############################################################################################
# #                                      Robust04 description                               #
#############################################################################################

# rel_ann_path = "/projects/0/guse0488/dataset/entities/trec-robust04/desc.query_annotations.tsv"
# ground_truth = {}
# with open(rel_ann_path, "r") as f:
#     for line in f:
#         qid, entities = line.strip().split("\t")
#         entities = json.loads(entities)
#         ground_truth[qid] = set()
#         for ent in entities:
#             ground_truth[qid].add(ent["entity_name"])

# entity_collection = set()
# num_candidates = []
# recalls = []
# cand_retriever = CandidateRetriever()
# entitydict = json.load(open("data/entity_list.json"))
# dataset = ir_datasets.load("disks45/nocr/trec-robust-2004")
# for query in tqdm(list(dataset.queries_iter())):
#     q_id, q_text = query.query_id, query.description
#     entity_ids, *_ = cand_retriever.retrieve_candidates(q_text)
#     entity_names = [entitydict[eid] for eid in entity_ids]
#     num_candidates.append(len(entity_ids))
#     entity_collection.update(entity_names)
#     if q_id in ground_truth and len(ground_truth[q_id]) > 0:
#         recall = len(set(entity_names).intersection(
#             ground_truth[q_id]))/len(ground_truth[q_id])
#         recalls.append(recall)
#     if len(recalls) > 20000:
#         break

# avg_num_cands = sum(num_candidates)/len(num_candidates)
# avg_recall = sum(recalls)/len(recalls)
# total_entities = len(entity_collection)
# print(f"Number of candidates per query: {avg_num_cands}")
# print(f"Recall per query: {avg_recall}")
# print(f"Total number of entities: {total_entities}")


#############################################################################################
# #                                      Codec query                                        #
#############################################################################################

# rel_ann_path = "/projects/0/guse0488/dataset/entities/codec/query_entity.json"
# ground_truth = json.load(open(rel_ann_path))
# for qid in ground_truth:
#     ground_truth[qid] = set(ground_truth[qid])

# entity_collection = set()
# num_candidates = []
# recalls = []
# cand_retriever = CandidateRetriever()
# entitydict = json.load(open("data/entity_list.json"))
# dataset = ir_datasets.load("codec")
# for query in tqdm(list(dataset.queries_iter())):
#     q_id, q_text = query.query_id, query.query
#     entity_ids, *_ = cand_retriever.retrieve_candidates(q_text)
#     entity_names = [entitydict[eid] for eid in entity_ids]
#     num_candidates.append(len(entity_ids))
#     entity_collection.update(entity_names)
#     if q_id in ground_truth and len(ground_truth[q_id]) > 0:
#         recall = len(set(entity_names).intersection(
#             ground_truth[q_id]))/len(ground_truth[q_id])
#         recalls.append(recall)
#     if len(recalls) > 20000:
#         break

# avg_num_cands = sum(num_candidates)/len(num_candidates)
# avg_recall = sum(recalls)/len(recalls)
# total_entities = len(entity_collection)
# print(f"Number of candidates per query: {avg_num_cands}")
# print(f"Recall per query: {avg_recall}")
# print(f"Total number of entities: {total_entities}")


#############################################################################################
# #                                      Codec document                                     #
#############################################################################################
from collections import defaultdict
rel_ann_path = "/projects/0/guse0488/dataset/entities/codec/codec_entity_links.jsonl"
ground_truth = defaultdict(set)
with open(rel_ann_path, "r") as f:
    for line in f:
        doc = json.loads(line)
        doc_id = list(doc.keys())[0]
        entities = []

        for mention in list(doc.values())[0]:
            entities.append(mention["prediction"].replace("_", " "))
        ground_truth[doc_id] = set(entities)

entity_collection = set()
num_candidates = []
recalls = []
cand_retriever = CandidateRetriever()
entitydict = json.load(open("data/entity_list.json"))
dataset = ir_datasets.load("codec")
for doc in tqdm(list(dataset.docs_iter())):
    # q_id, q_text = query.query_id, query.query
    d_text = doc.title + " " + doc.text
    entity_ids, *_ = cand_retriever.retrieve_candidates(d_text)
    entity_names = [entitydict[eid] for eid in entity_ids]
    num_candidates.append(len(entity_ids))
    entity_collection.update(entity_names)
    if doc.doc_id in ground_truth and len(ground_truth[doc.doc_id]) > 0:
        recall = len(set(entity_names).intersection(
            ground_truth[doc.doc_id]))/len(ground_truth[doc.doc_id])
        recalls.append(recall)
    if len(recalls) > 20000:
        break
avg_num_cands = sum(num_candidates)/len(num_candidates)
avg_recall = sum(recalls)/len(recalls)
total_entities = len(entity_collection)
print(f"Number of candidates per query: {avg_num_cands}")
print(f"Recall per query: {avg_recall}")
print(f"Total number of entities: {total_entities}")
