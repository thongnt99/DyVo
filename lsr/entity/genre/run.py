from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import ir_datasets
import json
import pickle
from tqdm import tqdm
from .trie import Trie
with open("kilt_titles_trie_dict.pkl", "rb") as f:
    trie = Trie.load_from_dict(pickle.load(f))
tokenizer = AutoTokenizer.from_pretrained("facebook/genre-kilt")
model = AutoModelForSeq2SeqLM.from_pretrained(
    "facebook/genre-kilt").eval().to("cuda")

dataset = ir_datasets.load("disks45/nocr/trec-robust-2004")
query_ids = []
query_texts = []
for query in dataset.queries_iter():
    query_ids.append(query.query_id)
    query_texts.append(query.description)

entity_dict = json.load(open("data/entity_list.json"))
entity_dict = dict(zip(entity_dict, list(range(len(entity_dict)))))
with open("robust04_desc_genre.jsonl", "w") as f:
    for qid, qtext in tqdm(zip(query_ids, query_texts)):
        outputs = model.generate(
            **tokenizer([qtext], padding=True, truncation=True, return_tensors="pt").to("cuda"),
            num_beams=5,
            num_return_sequences=5,
            prefix_allowed_tokens_fn=lambda batch_id, sent: trie.get(
                sent.tolist()),
        )
        entities = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        entity_ids = [entity_dict[ename]
                      for ename in entities if ename in entity_dict]
        obj = {"id": qid, "entity_names": entities, "entities": entity_ids}
        f.write(json.dumps(obj)+"\n")
