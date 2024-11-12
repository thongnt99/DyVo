from transformers import AutoModel, AutoTokenizer
import torch
import faiss
import json
from tqdm import tqdm


import json
import argparse
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
parser = argparse.ArgumentParser(
    "Encode queries and documents with DPR model trained from NQ")
parser.add_argument("--inp", type=str, help="Input path")
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
weight_path = "/projects/0/guse0488/lsr-entities-project/BLINK/models/biencoder_wiki_large.bin"
index_path = "/projects/0/guse0488/lsr-entities-project/BLINK_index/faiss_flat_index.pkl"
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
context_encoder = AutoModel.from_pretrained("bert-large-uncased")
print("Loading model checkpoint")
params = torch.load(weight_path, map_location="cpu")
params = {k.replace("context_encoder.bert_model.", ""): v for k, v in params.items(
) if "context_encoder.bert_model." in k}
context_encoder.load_state_dict(params)
context_encoder.to("cuda")
embeddings = []
for i in tqdm(range(0, len(texts), args.bs), desc="Encoding text"):
    text_batch = tokenizer(
        texts[i: i+args.bs], padding=True, truncation=True, return_tensors='pt', max_length=100).to("cuda")
    with torch.no_grad(), torch.cuda.amp.autocast():
        batch_embeddings = context_encoder(
            **text_batch)[0][:, 0, :]
        embeddings.append(batch_embeddings.to("cpu"))
embeddings = torch.cat(embeddings, dim=0).numpy()
df = pd.DataFrame({"id": ids, "emb": embeddings.tolist()})
df.to_parquet(args.out)
