import json
import re
import pandas as pd
from nltk.tokenize import word_tokenize
import numpy as np
import cstl
import json
import pickle


class SafeDict():
    def __init__(self, pydict: dict):
        key2offset = {}
        list_entities = []
        list_scores = []
        idx = 0
        for k, v in pydict.items():
            start_idx = idx
            entities, scores = list(zip(*v[:30]))
            list_entities.extend(entities)
            list_scores.extend(scores)
            idx = idx + len(entities)
            key2offset[k] = [start_idx, idx]
        self.list_entities = np.asarray(list_entities, dtype=np.int32)
        self.list_scores = np.asarray(list_scores, dtype=np.float16)
        self.key2offset = cstl.frompy(key2offset)

    def __len__(self):
        return len(self.key2offset)

    def __contains__(self, key):
        return key in self.key2offset

    def __getitem__(self, key):
        start, end = self.key2offset[key]
        return self.list_entities[start:end], self.list_scores[start:end]


class CandidateRetriever:
    def __init__(self, pme_path="data/wiki_p_m_e.json", embs_path="data/wiki_embs.pkl.npy") -> None:
        self.wiki_pme = SafeDict(json.load(open(pme_path)))
        self.ent_embs = np.load(embs_path)
        # self.ent_embs = torch.load(embs_path)
        # self.ent_embs = pd.read_parquet(
        #     embs_path)["emb"].tolist()

    def retrieve_candidates(self, text):
        mentions = self.generate_mention_candidates(text.lower())
        ent_candidates = []
        ent_scores = []
        for m in mentions:
            if m in self.wiki_pme:
                candidates, scores = self.wiki_pme[m]
                ent_candidates.extend(candidates)
                ent_scores.extend(scores)
        # sort, to only keep the largest scores
        if len(ent_candidates) == 0:
            return [], [], []
        pairs = sorted(list(zip(ent_candidates, ent_scores)),
                       key=lambda x: x[1])[-200:]
        e2cand = dict(pairs)
        entity_ids, entity_scores = list(zip(*e2cand.items()))
        entity_embs = [self.ent_embs[eid].tolist() for eid in entity_ids]
        return [list(entity_ids), entity_embs, list(entity_scores)]
        # return pairs

    def generate_mention_candidates(self, text):
        words = word_tokenize(text)
        mentions = set()
        for i in range(5):
            n_gram_len = i+1
            for j in range(len(words) - i):
                check = all([not re.match(r"^[_\W]+$", w)
                            for w in words[j:j+n_gram_len]])
                if check:
                    mention = " ".join(words[j:j+n_gram_len])
                    if len(mention) > 1:
                        mentions.add(mention)
        return list(mentions)
