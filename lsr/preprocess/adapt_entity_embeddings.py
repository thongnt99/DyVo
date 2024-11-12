from lsr.models.entity_emb import BertEntityEmbedding
from transformers import AutoTokenizer, AutoModel
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import json
from tqdm import tqdm
import argparse
parser = argparse.ArgumentParser("Adapt Entity Embedding")
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--fp16", action="store_true")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--inp", type=str, default="data/entity_bert_emb_agg.pt")
parser.add_argument("--out", type=str,
                    default="data/aligned_entity_bert_emb_model.pt")
args = parser.parse_args()


class EntityDataset(Dataset):
    def __init__(self, entity_list):
        super().__init__()
        self.entity_list = entity_list

    def __len__(self):
        return len(self.entity_list)

    def __getitem__(self, index):
        return index+30522, self.entity_list[index]


def ent_collate_fn(entities):
    entity_ids = []
    entity_names = []
    for ent_id, ent_name in entities:
        entity_ids.append(ent_id)
        entity_names.append(ent_name)
    return entity_ids, entity_names


model = BertEntityEmbedding(entity_emb_path=args.inp).to(args.device)
bert_static_emb = AutoModel.from_pretrained(
    "distilbert-base-uncased").embeddings.word_embeddings.to(args.device)
entity_list = json.load(open("data/entity_list.json"))
training_data = EntityDataset(entity_list)
train_dataloader = DataLoader(
    training_data, collate_fn=ent_collate_fn, batch_size=args.batch_size)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
ce_loss = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
for epoch_idx in range(args.epochs):
    acc_loss = 0
    for idx, batch_data in enumerate(tqdm(train_dataloader, desc=f"Training epoch: {epoch_idx}")):
        batch_ent_ids, batch_ent_names = batch_data
        optimizer.zero_grad()
        batch_ent_ids = torch.tensor(batch_ent_ids, device=args.device)
        tok_inps = tokenizer(batch_ent_names, return_tensors="pt",
                             padding=True, truncation=True).to(args.device)
        tok_embs = bert_static_emb(
            tok_inps["input_ids"])  # batch_size x L x 768
        out_embs = tok_embs.sum(
            dim=1) / tok_inps["attention_mask"].float().sum(dim=1).unsqueeze(-1)
        inp_embs = model(batch_ent_ids.unsqueeze(0)).squeeze(0)
        scores = inp_embs @ out_embs.T  # batch_size x batch_size
        scores1 = out_embs @ inp_embs.T
        labels = torch.arange(scores.size(0), device=args.device)
        loss = ce_loss(scores, labels) + ce_loss(scores1, labels)
        acc_loss += loss.item()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if idx > 0 and idx % 1000 == 0:
            print(f"Loss: {acc_loss}")
            acc_loss = 0
    print(f"Saving output model to: {args.out}")
    torch.save(model.state_dict(), args.out)
