from peft import PeftModel, PeftConfig
from transformers import AutoModel, AutoTokenizer
import torch
from tqdm import tqdm
import argparse
import json
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--inp", type=str)
parser.add_argument("--out", type=str)
parser.add_argument("--f", type=str, default="jsonl",
                    help="Input format: jsonl or tsv")
parser.add_argument("--batch_size", type=int, default=8)
args = parser.parse_args()


def get_model(peft_model_name):
    config = PeftConfig.from_pretrained(peft_model_name)
    base_model = AutoModel.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(base_model, peft_model_name)
    model = model.merge_and_unload()
    model.eval()
    return model


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

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
tokenizer.pad_token = tokenizer.eos_token
model = get_model('castorini/repllama-v1-7b-lora-passage').to("cuda")

for i in range(len(texts)):
    texts[i] = f"passage: {texts[i]}"


def encode(texts):
    embs = []
    for idx in tqdm(range(0, len(texts), args.batch_size), desc="Encoding"):
        with torch.no_grad(), torch.cuda.amp.autocast():
            inputs = tokenizer(texts[idx: idx+args.batch_size],
                               padding=False, truncation=True, max_length=511)
            batch_size = len(inputs["input_ids"])
            batch_max_len = 0
            for i in range(batch_size):
                inputs["input_ids"][i].append(tokenizer.eos_token_id)
                inputs["attention_mask"][i].append(1)
                batch_max_len = max(batch_max_len, len(inputs["input_ids"][i]))
            for i in range(batch_size):
                num_pad_token = (batch_max_len - len(inputs["input_ids"][i]))
                inputs["input_ids"][i] += [0] * num_pad_token
                inputs["attention_mask"][i] += [0] * num_pad_token
            inputs["input_ids"] = torch.tensor(inputs["input_ids"]).to("cuda")
            inputs["attention_mask"] = torch.tensor(
                inputs["attention_mask"]).to("cuda")
            length = inputs["attention_mask"].sum(dim=1) - 1
            batch_embs = model(
                **inputs).last_hidden_state[torch.arange(batch_size), length]
            batch_embs = torch.nn.functional.normalize(batch_embs, p=2, dim=1)
            embs.append(batch_embs)
    return torch.cat(embs, dim=0)


embs = encode(texts)
df = pd.DataFrame({"id": ids, "emb": embs.tolist()})
df.to_parquet(args.out)
