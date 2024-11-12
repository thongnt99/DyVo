from collections import defaultdict
import ir_measures
import json
qid2text = {}
with open("data/msmarco/queries.tsv", "r") as f:
    for line in f:
        qid, text = line.strip().split("\t")
        qid2text[qid] = text
entity_dict = json.load(open("data/entity_list.json"))

run = ir_measures.read_trec_run("entity_retrieval/bm25/msmarco_queries.trec")

res = defaultdict(list)
for row in run:
    entity = entity_dict[int(row.doc_id)]
    res[row.query_id].append(entity)

with open("entity_retrieval/msmaro_queries.tsv", "w") as f:
    for qid in res:
        f.write(f"qid: {qid} text: {qid2text[qid]}:  ")
        for ent in res[qid]:
            f.write(ent+"\n")
        f.write("\n")

rel_ann_path = "/projects/0/guse0488/dataset/mmead/topics.msmarco-passage.dev-subset.linked.json"
ground_truth = {}
with open(rel_ann_path, "r") as f:
    for line in f:
        query = json.loads(line.strip())
        ground_truth[query["qid"]] = set()
        for ent in query["links"]:
            ground_truth[query["qid"]].add(ent["entity"])

recalls = []
for qid in ground_truth:
    if len(ground_truth[qid]) == 0:
        continue
    bm25_entities = set(res[qid])
    recalls.append(
        len(bm25_entities.intersection(ground_truth[qid]))*1.0/len(ground_truth[qid]))
avg_recall = sum(recalls)/len(recalls)
print(f"Recall per query: {avg_recall}")
