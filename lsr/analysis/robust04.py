import ir_datasets
import irds_robust_anserini

dataset = ir_datasets.load("macavaney:anserini-trec-robust04")
with open("data/robust04/queries.tsv", "w") as f:
    for queries in dataset.queries_iter():
        f.write(f"{queries.query_id}\t{ queries.description}\n")

with open("data/robust04/docs.tsv", "w") as f:
    for doc in dataset.docs_iter():
        text = doc.text.replace("\n", "")
        f.write(f"{doc.doc_id}\t{text}\n")
