from transformers import AutoModelForMaskedLM, AutoTokenizer, AutoModel
import torch
import json
import faiss
from faiss import read_index

weight_path = "../BLINK/models/biencoder_wiki_large.bin"
index_path = "data/blink_hnsw_flat.index"
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
context_encoder = AutoModelForMaskedLM.from_pretrained(
    "lsr42/blink_ctx_encoder")
# context_encoder = AutoModelForMaskedLM.from_pretrained("lsr42/elq_ctx_encoder")
text = "what is tesla's net worth"
# text = "how much can i contribute to nondeductible ira"
inps = tokenizer(text, return_tensors="pt")
tokens = tokenizer.convert_ids_to_tokens(inps["input_ids"][0].tolist())
context_encoder.eval()
with torch.no_grad():
    outputs = context_encoder(**inps, output_hidden_states=True)
    # print(outputs)
    hidden_states = outputs.hidden_states
    last_hidden_state = hidden_states[-1][0].detach()
    # last_hid
index = read_index(index_path)
index.efSearch = 128
dis, indices = index.search(last_hidden_state, 10)
entity_list = json.load(open("data/entity_list.json"))
for idx, tok in enumerate(tokens):
    print(tok)
    for eid, score in zip(indices[idx], dis[idx]):
        print(entity_list[eid], " ", int(score*100)/100.0)
mlm_logits = outputs.logits[0]
values, indices = mlm_logits.max(dim=-1)
print(tokens)
print(tokenizer.convert_ids_to_tokens(indices.tolist()))
print(values.tolist())
