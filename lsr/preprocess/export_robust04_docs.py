import ir_datasets

data = ir_datasets.load("disks45/nocr/trec-robust-2004")
with open("data/robust04/docs.tsv", "w") as f:
    for doc in data.docs_iter():
        text = (doc.title + " " +
                doc.body).replace("\n", " ").replace("\t", " ").strip()
        if text:
            f.write(f"{doc.doc_id}\t{text}\n")
