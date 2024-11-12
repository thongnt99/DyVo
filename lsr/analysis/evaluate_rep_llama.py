from datasets import load_dataset
from peft import PeftModel, PeftConfig
from collections import defaultdict
from transformers import AutoModel, AutoTokenizer
import ir_datasets
import ir_measures
from ir_measures import *
import torch
from tqdm import tqdm
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str,
                    default="disks45/nocr/trec-robust-2004")
parser.add_argument("--batch_size", type=int, default=32)
args = parser.parse_args()

queries = []
query_ids = []
# doc_ids = []
# docs = []


def get_model(peft_model_name):
    config = PeftConfig.from_pretrained(peft_model_name)
    base_model = AutoModel.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(base_model, peft_model_name)
    model = model.merge_and_unload()
    model.eval()
    return model


dataset = ir_datasets.load(args.dataset)
for query in dataset.queries_iter():
    query_ids.append(query.query_id)
    queries.append(query.description)

# for doc in dataset.docs_iter():
#     doc_ids.append(doc.doc_id)
#     docs.append(str(doc.title) + " " + doc.body)

tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
tokenizer.pad_token = tokenizer.eos_token
model = get_model('castorini/repllama-v1-7b-lora-passage').to("cuda")


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


for i in range(len(queries)):
    queries[i] = f"query: {queries[i]}</s>"
query_embs = encode(queries)
# doc_embs = encode(docs)
docs_data = load_dataset("parquet", data_files={
    "entities": "data/robust04/docs_repllama/*.parquet"}, keep_in_memory=True).with_format("torch")
doc_ids = docs_data["entities"]["id"]
doc_embs = docs_data["entities"]["emb"].to("cuda")

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
