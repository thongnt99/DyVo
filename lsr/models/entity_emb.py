from torch import nn
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
import json
import faiss
from faiss import read_index
import numpy as np


class BertEntityEmbedding(nn.Module):
    def __init__(self, *args, entity_emb_path="data/entity_bert_emb_agg.pt", ** kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.entity_emb = torch.load(
            entity_emb_path, map_location="cpu").to("cpu")

    def forward(self, entity_ids):
        with torch.no_grad():
            batch_size, num_ent = entity_ids.size()
            indices = (entity_ids.view(-1)-30522).to("cpu")
            entity_embs = self.entity_emb[indices].to(entity_ids.device)
            entity_embs = entity_embs.view(batch_size, num_ent, -1)
        # entity_embs = self.entity_projection(entity_embs)
        return entity_embs

    # def search_and_reconstruct(self, queries):
    #     # queries = queries.
    #     # dis = queries.to("cpu") @ self.entity_emb.T
    #     # top100_scores, top100_ids = torch.topk(dis, k=100)
    #     # top100_ids = top100_ids + 30522
    #     _, top100_ids, top100_ent_embs = self.hnsw_index.search_and_reconstruct(
    #         queries.detach().to("cpu"), 100)
    #     top100_ent_embs = np.nan_to_num(top100_ent_embs)
    #     top100_ent_ids = torch.tensor(
    #         top100_ids + 30522, device=queries.device)
    #     top100_ent_embs = torch.from_numpy(
    #         top100_ent_embs).to(queries.device)
    #     return top100_ent_ids, top100_ent_embs
        # return top100_scores.to(queries.device), top100_ids.to(queries.device)
