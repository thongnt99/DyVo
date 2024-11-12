from transformers import AutoModel, AutoTokenizer
import json
import torch
from tqdm import tqdm
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
emb_model = AutoModel.from_pretrained(
    "distilbert-base-uncased").embeddings.word_embeddings.to("cuda")
entity_list = json.load(open("data/entity_list.json"))
batch_size = 1024
embs = []
for idx in tqdm(range(0, len(entity_list), batch_size)):
    batch_entity = entity_list[idx: idx + batch_size]
    batch_entity_tokens = tokenizer(
        batch_entity, padding=True, truncation=True, return_tensors="pt").to("cuda")
    with torch.no_grad(), torch.cuda.amp.autocast():
        token_embs = emb_model(
            batch_entity_tokens["input_ids"]) * batch_entity_tokens["attention_mask"].unsqueeze(-1)
        batch_entity_embs = token_embs.sum(
            dim=1) / batch_entity_tokens["attention_mask"].sum(dim=1).unsqueeze(-1)
    embs.append(batch_entity_embs)
embs = torch.cat(embs, dim=0)
torch.save(embs, "data/entity_bert_emb_agg.pt")
