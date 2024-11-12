import json
import argparse
import pandas as pd
import torch
from tqdm import tqdm
import torch
from sentence_transformers import SentenceTransformer

parser = argparse.ArgumentParser(
    "Encode queries and documents with DPR model trained from NQ")
parser.add_argument("--inp", type=str, help="Input path")
parser.add_argument("--model", type=str,
                    default="lsr42/distilbert-base-uncased-dense-laque")
parser.add_argument("--out", type=str, help="Output path")
parser.add_argument("--f", type=str, default="jsonl",
                    help="Input format: jsonl or tsv")
parser.add_argument("--bs", type=int, default=256, help="batch size")
parser.add_argument("--topk", type=int, default=20)

args = parser.parse_args()

texts = []
ids = []
if args.f == "jsonl":
    with open(args.inp, "r") as f:
        for line in f:
            data = json.loads(line.strip())
            ids.append(data["id"])
            texts.append(data["contents"])
else:
    with open(args.inp, "r") as f:
        for line in f:
            try:
                qid, qtext = line.strip().split("\t")
                ids.append(qid)
                texts.append(qtext)
            except:
                print(line)
model = SentenceTransformer(args.model)
text_embeddings = torch.tensor(model.encode(texts))

laque_entity_embs = torch.load("data/entity_embs_laque.pt")
topk_entities = []
for idx in tqdm(range(0, len(ids), args.bs), desc="Scoring entities"):
    batch_embs = text_embeddings[idx: idx+args.bs]
    dis = batch_embs @ laque_entity_embs.T
    _score, entity_ids = dis.topk(args.topk, dim=1)
    topk_entities.extend(entity_ids.tolist())
assert len(topk_entities) == len(ids)
with open(args.out, "w") as f:
    for text_id, entity_cands in zip(ids, topk_entities):
        data = {"id": text_id, "entities": entity_cands}
        f.write(json.dumps(data)+"\n")
