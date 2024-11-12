from datasets import load_dataset
import torch
docs_data = load_dataset("parquet", data_files={
    "entities": "data/entity_embs_laque/*.parquet"}, keep_in_memory=True).with_format("torch")
entity_embs = torch.zeros(5277580, 768)
ids = docs_data["entities"]["id"]
embs = docs_data["entities"]["emb"]
for eidx, e_emb in zip(ids, embs):
    entity_embs[eidx] = e_emb
torch.save(entity_embs, "entity_embs_laque.pt")
