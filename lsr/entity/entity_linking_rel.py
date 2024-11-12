import json
from REL.mention_detection import MentionDetection
from REL.utils import process_results
from REL.entity_disambiguation import EntityDisambiguation
from REL.ner import Cmns, load_flair_ner
import argparse
parser = argparse.ArgumentParser("Argument parser")
parser.add_argument("--inp", type=str,
                    default="data/wapo/inparsv2/topic-wapo.tsv", help="Input")
parser.add_argument(
    "--out", type=str, default="data/wapo/inparsv2/queries-entities-flair-ner.json")
args = parser.parse_args()
wiki_version = "wiki_2019"
base_url = "/projects/0/guse0488/lsr-entities-project/lsr-entities/rel/data"
mention_detection = MentionDetection(base_url, wiki_version)
tagger_ngram = Cmns(base_url, wiki_version, n=5)
# tagger_ner = load_flair_ner("ner-fast")
config = {
    "mode": "eval",
    "model_path": "ed-wiki-2019",
}
model = EntityDisambiguation(base_url, wiki_version, config)

input_queries = {}
with open(args.inp, "r") as f:
    for line in f:
        qid, qtext = line.strip().split("\t")
        input_queries[qid] = (qtext, [])
mentions, n_mentions = mention_detection.find_mentions(
    input_queries, tagger_ngram)
predictions, timing = model.predict(mentions)
rel_entities = process_results(mentions, predictions, input_queries)
# json.dump(result, open(args.out, "w"))
entity_list = json.load(open("data/entity_list.json", "r"))
entity2id = {ent_name: idx for idx, ent_name in enumerate(entity_list)}
# rel_entities = json.load(open(input_path, "r"))
with open(args.out, "w") as f:
    for qid in rel_entities:
        entity_names = [entry[3] for entry in rel_entities[qid]]
        entity_ids = [entity2id[ent_name.replace(
            "_", " ")] for ent_name in entity_names if ent_name.replace(
            "_", " ") in entity2id]
        entity_ids = list(set(entity_ids))
        qentities = {"id": qid, "entities": entity_ids}
        f.write(json.dumps(qentities)+"\n")
