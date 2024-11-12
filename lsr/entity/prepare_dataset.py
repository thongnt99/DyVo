import ir_datasets
import json

dataset = ir_datasets.load("msmarco-passage/train")
with open("msmarco_train_queries.jsonl", "w") as f:
    for query in dataset.queries_iter():
        q_id, q_text = query.query_id, query.text
        data = {"qid": q_id, "text": q_text}
        f.write(json.dumps(data)+"\n")
