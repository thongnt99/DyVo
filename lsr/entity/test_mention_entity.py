
from transformers import AutoTokenizer
from REL.entity_disambiguation import EntityDisambiguation
from .mention_detection import MentionDetection
from REL.utils import process_results
from REL.ner import Cmns, load_flair_ner
base_url = "/projects/0/guse0488/lsr-entities-project/lsr-entities/rel/data"
wiki_version = "wiki_2019"
tagger_ngram = Cmns(base_url, wiki_version, n=5)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
example = {"1": [
    "is USA part of WHO ?", []]}
mention_detection = MentionDetection(base_url, wiki_version)
mentions_dataset, n_mentions = mention_detection.find_mentions(
    example, tagger_ngram)
for doc_id in mentions_dataset:
    for mention in mentions_dataset[doc_id]:
        print(mention["mention"])
        print(mention["candidates"])
        print(len(mention["candidates"]))
# print(mentions_dataset)
