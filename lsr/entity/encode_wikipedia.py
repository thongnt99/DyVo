import json
import argparse
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
parser = argparse.ArgumentParser(
    "Encode queries and documents with DPR model trained from NQ")
parser.add_argument("--inp", type=str, help="Input path")
parser.add_argument("--type", type=str, default="query",
                    help="Input type: query, doc")
parser.add_argument("--out", type=str, help="Output path")
parser.add_argument("--f", type=str, default="jsonl",
                    help="Input format: jsonl or tsv")
parser.add_argument("--bs", type=int, default=256, help="batch size")
args = parser.parse_args()


def cls_pooling(model_output, attention_mask):
    return model_output[0][:, 0]


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
if args.type == "query":
    tokenizer = AutoTokenizer.from_pretrained(
        'sentence-transformers/facebook-dpr-question_encoder-single-nq-base')
    model = AutoModel.from_pretrained(
        'sentence-transformers/facebook-dpr-question_encoder-single-nq-base').to("cuda")
else:
    tokenizer = AutoTokenizer.from_pretrained(
        'sentence-transformers/facebook-dpr-ctx_encoder-single-nq-base')
    model = AutoModel.from_pretrained(
        'sentence-transformers/facebook-dpr-ctx_encoder-single-nq-base').to("cuda")
embeddings = []
for i in tqdm(range(0, len(texts), args.bs), desc="Encoding text"):
    text_batch = tokenizer(
        texts[i: i+args.bs], padding=True, truncation=True, return_tensors='pt', max_length=100).to("cuda")
    with torch.no_grad(), torch.cuda.amp.autocast():
        model_output = model(**text_batch)
        batch_embeddings = cls_pooling(
            model_output, text_batch['attention_mask'])
        embeddings.append(batch_embeddings.to("cpu"))

embeddings = torch.cat(embeddings, dim=0).numpy()
df = pd.DataFrame({"id": ids, "emb": embeddings.tolist()})
df.to_parquet(args.out)
