from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForMaskedLM
import ir_datasets
import ir_measures
from ir_measures import *
import torch
import tqdm 
import torch.nn.functional as F
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str,
                    default="disks45/nocr/trec-robust-2004")
parser.add_argument("--model", type=str, default='naver/splade-v3-distilbert')
parser.add_argument("--query_field", type=str, default="description")
parser.add_argument("--doc_fields", type=str, nargs="+", default="description")
parser.add_argument("--cos_sim", action="store_true")
args = parser.parse_args()

queries = []
query_ids = []
doc_ids = []
docs = []

dataset = ir_datasets.load(args.dataset)
for query in dataset.queries_iter():
    query_ids.append(query.query_id)
    queries.append(getattr(query, args.query_field))

for doc in dataset.docs_iter():
    doc_ids.append(doc.doc_id)
    doc_text = [str(getattr(doc, field)) for field in args.doc_fields]
    doc_text = " ".join(doc_text)
    docs.append(doc_text)

# Load model directly

tokenizer = AutoTokenizer.from_pretrained("naver/splade-v3-distilbert")
model = AutoModelForMaskedLM.from_pretrained("naver/splade-v3-distilbert").to("cuda")

def encode(texts, batch_size=256):
    len_n = len(texts)
    all_reps = []
    for i in tqdm(range(0, len_n, batch_size), desc="Splade-v3 encoding"):
        with torch.no_grad(), torch.cuda.amp.autocast():
            batch_texts = texts[i: i + batch_size]
            inps = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True).to("cuda")
            output = model(**inps)
            reps = output.logits.max(dim=1).values
            reps = torch.log1p(torch.relu(reps))
            all_reps.append(reps)
    all_reps = torch.cat(all_reps, dim=0)
    return all_reps


# model = SentenceTransformer(args.model)

# with torch.cuda.amp.autocast():
# query_embs = model.encode(queries, show_progress_bar=True,
#                             batch_size=128, convert_to_tensor=True)
# doc_embs = model.encode(docs, show_progress_bar=True,
#                         batch_size=128, convert_to_tensor=True)

# if args.cos_sim:
#     query_embs = F.normalize(query_embs, p=2, dim=1)
#     doc_embs = F.normalize(doc_embs, p =2, dim=1)
query_embs = encode(queries)
doc_embs = encode(docs)

scores = query_embs @ doc_embs.T

topk_scores, topk_indices = scores.topk(k=1000, dim=1)


run = {}
for idx, q_id in enumerate(query_ids):
    topdoc_ids = [doc_ids[i] for i in topk_indices[idx].tolist()]
    run[q_id] = dict(zip(topdoc_ids, topk_scores[idx].tolist()))

qrels = list(dataset.qrels_iter())

metrics = ir_measures.calc_aggregate(
    [NDCG@10, NDCG@20, R@100, R@1000], qrels, run)
print(metrics)
