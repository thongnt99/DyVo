import tqdm
import json
output_path = "/projects/0/guse0488/dataset/entities/trec-robust04/queries_inparsv2_ngram.jsonl"
input_path = "data/robust04/inparsv2/queries-entities-rel-ngram.json"

entity_list = json.load(open("data/entity_list.json", "r"))
entity2id = {ent_name: idx for idx, ent_name in enumerate(entity_list)}
rel_entities = json.load(open(input_path, "r"))
with open(output_path, "w") as f:
    for qid in rel_entities:
        entity_names = [entry[3] for entry in rel_entities[qid]]
        entity_ids = [entity2id[ent_name.replace(
            "_", " ")] for ent_name in entity_names if ent_name.replace(
            "_", " ") in entity2id]
        entity_ids = list(set(entity_ids))
        qentities = {"id": "inparsv2_"+qid, "entities": entity_ids}
        f.write(json.dumps(qentities)+"\n")

# monot5scores = json.load(open("data/robust04/inparsv2/monot5_3b_scores.json"))
# prefix_monot5scores = {"inparsv2_"+qid: scores for qid,
#                        scores in monot5scores.items()}
# json.dump(prefix_monot5scores, open(
#     "data/robust04/inparsv2/prefix_monot5_3b_scores.json", "w"))
# with open("data/robust04/inparsv2/topics-robust04.tsv", "r") as fin, open("data/robust04/inparsv2/prefix_topics-robust04.tsv", "w") as fout:
#     for line in fin:
#         qid, qtext = line.strip().split("\t")
#         qid = "inparsv2_" + qid
#         fout.write(f"{qid}\t{qtext}\n")

# with open("data/robust04/inparsv2/topics-robust04.tsv", "r") as fin, open("data/robust04/inparsv2/prefix_topics-robust04.tsv", "w") as fout:
#     for line in fin:
#         qid, qtext = line.strip().split("\t")
#         qid = "inparsv2_" + qid
#         fout.write(f"{qid}\t{qtext}\n")
# qrels = json.load(open("data/robust04/inparsv2/robust04_qrels.json", "r"))
# prefix_qrels = {"inparsv2_"+qid: rels for qid, rels in qrels.items()}
# json.dump(prefix_qrels, open(
#     "data/robust04/inparsv2/prefix_robust04_qrels.json", "w"))
