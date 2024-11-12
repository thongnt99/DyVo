from .mention_detection import MentionDetection
from .ngram import Cmns
base_url = "/projects/0/guse0488/lsr-entities-project/lsr-entities/rel/data"
wiki_version = "wiki_2019"


class CandidateRetriever:
    def __init__(self, base_url="/projects/0/guse0488/lsr-entities-project/lsr-entities/rel/data", wiki_version="wiki_2019") -> None:
        self.base_url = base_url
        self.wiki_version = wiki_version
        self.md = MentionDetection(base_url, wiki_version)
        self.ngram_generators = Cmns(base_url, wiki_version)

    def get_candidates(self, text):
        ents, ent_ids, ent_embs, ent_scores = self.md.find_mentions(
            text, self.ngram_generators)
        return ents, ent_ids, ent_embs, ent_scores
