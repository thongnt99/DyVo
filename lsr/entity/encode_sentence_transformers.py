import json
import argparse
import pandas as pd
import torch
from tqdm import tqdm
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
embeddings = model.encode(texts)
df = pd.DataFrame({"id": ids, "emb": embeddings.tolist()})
df.to_parquet(args.out)
