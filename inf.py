from transformers import PretrainedConfig, PreTrainedModel, AutoModel, AutoModelForMaskedLM
from torch import nn
import torch
from transformers import AutoTokenizer
import ir_datasets
from tqdm import tqdm
import json
import argparse


class EPICConfig(PretrainedConfig):
    model_type = "lsr_epic"

    def __init__(self, activation="softplus", backbone="distilbert-base-uncased", ** kwargs):
        super().__init__(**kwargs)
        self.activation = activation
        self.backbone = backbone


class EPICQueryEncoder(PreTrainedModel):
    config_class = EPICConfig

    def __init__(self, config: EPICConfig = EPICConfig()):
        super().__init__(config)
        self.config = config
        self.model = AutoModel.from_pretrained(config.backbone)
        self.linear = nn.Linear(self.model.config.dim, 1)
        self.activation = nn.Softplus()

    def forward(self, input_ids, attention_mask, special_tokens_mask):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)
        tok_weights = self.linear(
            output.last_hidden_state).squeeze(-1)  # bs x len x 1
        tok_weights = (
            torch.log1p(self.activation(tok_weights))
            * attention_mask
            * (1 - special_tokens_mask)
        )
        return input_ids, tok_weights


class EPICDocEncoder(PreTrainedModel):
    config_class = EPICConfig

    def __init__(self, config: EPICConfig = EPICConfig()):
        super().__init__(config)
        self.config = config
        self.model = AutoModelForMaskedLM.from_pretrained(config.backbone)
        self.term_importance = nn.Sequential(
            nn.Linear(self.model.config.dim, 1),
            nn.Softplus())
        self.doc_importance = nn.Sequential(
            nn.Linear(self.model.config.dim, 1), nn.Sigmoid())
        self.activation = nn.Softplus()

    def forward(self, input_ids, attention_mask, special_tokens_mask, topk=400):
        output = self.model(
            input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_states = output.hidden_states[-1]
        term_scores = self.term_importance(last_hidden_states)
        cls_toks = output.hidden_states[-1][:, 0, :]
        doc_scores = self.doc_importance(cls_toks)
        logits = (
            output.logits
            * attention_mask.unsqueeze(-1)
            * (1 - special_tokens_mask).unsqueeze(-1)
            * term_scores
        )
        logits = torch.log1p(self.activation(logits))
        logits = torch.max(logits, dim=1).values * doc_scores
        max_non_zero = (logits > 0).sum(dim=-1).max()
        topk = min(topk, max_non_zero)
        tok_weights, tok_ids = torch.topk(logits, k=topk, dim=-1)
        return tok_ids, tok_weights


def encode_text(model, tokenizer,  id_list, text_list, output_file, batch_size):
    with open(output_file, "w") as f:
        for idx in tqdm(range(0, len(id_list), batch_size), desc=f"Encoding text and saving ouptut to {output_file}"):
            batch_ids = id_list[idx: idx+batch_size]
            batch_texts = text_list[idx: idx+batch_size]
            tokenized_batch = tokenizer(
                batch_texts, padding=True, truncation=True, return_special_tokens_mask=True, max_length=250, return_tensors="pt").to("cuda")
            with torch.no_grad(), torch.cuda.amp.autocast():
                batch_tok_ids, batch_tok_weights = model(**tokenized_batch)
                batch_tok_ids = batch_tok_ids.to("cpu").tolist()
                batch_tok_weights = batch_tok_weights.to("cpu").tolist()
            for text_id, text, tok_id_list, tok_weight_list in zip(batch_ids, batch_texts,  batch_tok_ids, batch_tok_weights):
                tok_list = tokenizer.convert_ids_to_tokens(tok_id_list)
                lex_rep = {"id": text_id,  "vector": {
                    tok: w for tok, w in zip(tok_list, tok_weight_list) if w > 0}}
                f.write(json.dumps(lex_rep)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Parsing arguments")
    parser.add_argument("--inp", type=str)
    parser.add_argument("--out", type=str)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    # query_encoder = EPICQueryEncoder.from_pretrained(
    #     "lsr42/ep_query_encoder").to("cuda")
    doc_encoder = EPICDocEncoder.from_pretrained(
        "lsr42/ep_doc_encoder").to("cuda")
    # data = ir_datasets.load("msmarco-passage/dev/small")
    batch_size = 1000
    # query_ids = []
    # query_texts = []
    # for query in tqdm(data.queries_iter(), desc="Loading queries"):
    #     query_ids.append(query.query_id)
    #     query_texts.append(query.text)
    # query_output_file = "msmarco-passage/epic/queries.tsv"
    # encode_text(query_encoder, tokenizer, query_ids,
    #             query_texts, query_output_file, batch_size=1000)
    passage_ids = []
    passage_texts = []
    with open(args.inp, "r") as f:
        for line in f:
            tid, text = line.strip().split("\t")
            passage_ids.append(tid)
            passage_texts.append(text)
    # for psg in tqdm(data.docs_iter(), desc="Loading documents"):
    #     passage_ids.append(psg.doc_id)
    #     passage_texts.append(psg.text)
    # psg_output_file = "msmarco-passage/epic/psgs.jsonl"
    encode_text(doc_encoder, tokenizer, passage_ids,
                passage_texts, args.out, batch_size=256)
