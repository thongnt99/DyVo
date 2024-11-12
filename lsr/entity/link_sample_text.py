
from REL.entity_disambiguation import EntityDisambiguation
from .mention_detection import MentionDetection
from REL.utils import process_results
from REL.ner import Cmns, load_flair_ner
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
input_dataset = {}
input_dataset[1] = ["is us a member of who?", []]
mentions_dataset, n_mentions = mention_detection.find_mentions(
    input_dataset, tagger_ner)
model = EntityDisambiguation(base_url, wiki_version, config)
predictions, timing = model.predict(mentions_dataset)
result = process_results(mentions_dataset, predictions, input_dataset)
print(result)
