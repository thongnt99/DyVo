from http.server import HTTPServer

from REL.entity_disambiguation import EntityDisambiguation
from .mention_detection import MentionDetection
from REL.utils import process_results
from REL.ner import Cmns, load_flair_ner
from REL.server import make_handler
import ir_datasets
from tqdm import tqdm
import json
base_url = "/projects/0/guse0488/lsr-entities-project/lsr-entities/rel/data"
wiki_version = "wiki_2019"
config = {
    "mode": "eval",
    # or alias, see also tutorial 7: custom models
    "model_path": "ed-wiki-2019",
}
mention_detection = MentionDetection(
    base_url=base_url, wiki_version=wiki_version)
tagger_ner = load_flair_ner("ner-fast-with-lowercase")
dataset = ir_datasets.load("msmarco-passage/train")
input_dataset = {}
for query in dataset.queries_iter():
    q_id, q_text = query.query_id, query.text
    input_dataset[q_id] = [q_text, []]
print(f"Loading queries done. {len(input_dataset)} queries are loaded")
mentions_dataset, n_mentions = mention_detection.find_mentions(
    input_dataset, tagger_ner)

model = EntityDisambiguation(base_url, wiki_version, config)
predictions, timing = model.predict(mentions_dataset)
result = process_results(mentions_dataset, predictions, input_dataset)
json.dump(result, open(
    "/projects/0/guse0488/dataset/mmead/topics.msmarco-passage.train.linked.json", "w"))
