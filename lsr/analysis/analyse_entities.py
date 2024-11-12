import numpy as np
import ir_datasets
import json
from tqdm import tqdm
doc_path = "outputs/qmlp_dmlm_emlm_msmarco_hn_l1_0.0_0.0001/11-11-2023/model/codec/tmp_eval/docs/docs_0.jsonl"
dataset = ir_datasets.load("codec")
entities = json.load(open("data/entity_list.json"))
with open(doc_path, "r") as f:
    for line in f:
        doc = json.loads(line)
        doc_id = doc["id"]
        tok2w = doc["vector"]
        doc_raw = dataset.docs_store().get(doc_id)
        print(doc_raw.title)
        print(doc_raw.text)
        print("==================================")
        output = ""
        for tok in tok2w:
            tok_id = int(tok)
            if tok_id >= 30522:
                ent_id = tok_id - 30522
                output = output + f"{entities[ent_id]}: {tok2w[tok]} | "
        print(output)
        input()

# doc_path = "outputs/qmlp_dmlm_emlm_msmarco_distil_l1_0.0_0.0001/10-17-2023/robust04/tmp_eval/queries.tsv"
# output_path = "analysis/robust04_entities.txt"
# entity_list = json.load(open("data/entity_list.json"))
# # dataset = ir_datasets.load("msmarco-passage")
# # doc_store = {doc.doc_id: doc.text for doc in tqdm(dataset.docs_iter())}
# doc_store = {}
# with open("data/robust04/docs.tsv") as f:
#     for line in tqdm(f):
#         doc_id, text = line.strip().split("\t")
#         doc_store[doc_id] = text.replace("\n", "")
# num_entities = []
# with open(doc_path, "r") as f, open(output_path, "w") as fout:
#     for line in f:
#         # data = json.loads(line.strip())
#         qid, qtext = line.split("\t")
#         from collections import Counter
#         data = {"vector":  Counter(qtext.split(" "))}
#         # doc_id = data["id"]
#         entities = []
#         for tok_id in data['vector']:
#             if int(tok_id) >= 30522:
#                 entities.append(
#                     {entity_list[int(tok_id)]: data['vector'][tok_id]})
#         num_entities.append(len(entities))
#         if len(entities) > 0:
#             fout.write(doc_id+"\n")
#             # fout.write(doc_store[doc_id]+"\n")
#             fout.write(json.dumps(entities))
#             fout.write("-----------------------")
# num_entities = np.array(num_entities)
# non_zeros = (num_entities > 0).sum()
# print(
#     f"Number of docs with entities: {non_zeros}/{len(num_entities)} - {non_zeros/len(num_entities)*100}")
# print(f"Average entities per doc: {np.mean(num_entities[num_entities>0])}")
