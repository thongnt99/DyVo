from segtok.segmenter import split_single
from collections import defaultdict
from REL.mention_detection_base import MentionDetectionBase
import numpy as np
from functools import lru_cache
import time


class MentionDetection(MentionDetectionBase):
    """
    Class responsible for mention detection.
    """

    def __init__(self, base_url, wiki_version):
        self.cnt_exact = 0
        self.cnt_partial = 0
        self.cnt_total = 0
        super().__init__(base_url, wiki_version)
        # self.create_index()

    def create_index(self):
        print("Start creating index")
        start = time.time()
        wikiIndex = "CREATE INDEX if not exists idx_{} ON {}({})".format(
            "word", "wiki", "word"
        )
        embIndex = "CREATE INDEX if not exists idx_{} ON {}({})".format(
            "emb", "embeddings", "emb"
        )
        self.wiki_db.cursor.execute(wikiIndex)
        self.wiki_db.cursor.execute(embIndex)
        end = time.time()
        print(f"Finish indexing. Time: {end-start}")

    def lookup_many(self, column, table_name, w):
        qmarks = ','.join(('?',)*len(w))
        return self.wiki_db.cursor.execute(
            f"select ROWID,word,{column} from {table_name} where word in ({qmarks})",
            w,
        ).fetchall()

    def lookup_list(self, w, table_name, column="emb"):
        """
        Args:
            w: list of words to look up.
        Returns:
            embeddings for ``w``, if it exists.
            ``None``, otherwise.
        """
        w = list(w)
        if len(w) == 0:
            return [], [], []
        else:
            ret = self.lookup_many(column, table_name, w)
            mapping = {ent_name: (ent_id, np.frombuffer(ent_emb, dtype=np.float32).tolist())
                       for ent_id, ent_name, ent_emb in ret}
            ent_names = []
            ent_ids = []
            ent_embs = []
            for ent in w:
                if ent in mapping:
                    ent_names.append(ent)
                    ent_ids.append(mapping[ent][0])
                    ent_embs.append(mapping[ent][1])
            return ent_names, ent_ids, ent_embs

    def find_mentions(self, raw_text, tagger=None):
        """
        Responsible for finding mentions given a set of documents in a batch-wise manner. More specifically,
        it returns the mention, its left/right context and a set of candidates.
        :return: Dictionary with mentions per document.
        """
        entity2score = defaultdict(lambda: 0)
        mentions = tagger.predict(raw_text)
        # print(len(mentions))
        for entity in mentions:
            mention_text, start_pos, end_pos, conf, tag = (
                entity.text,
                entity.start_position,
                entity.end_position,
                entity.score,
                entity.tag,
            )
            m = self.preprocess_mention(mention_text)
            cands = self.get_candidates(m)[:30]
            # if len(cands) == 0:
            #     continue
            # m2e = {
            #     "mention": m,
            #     "candidates": cands,
            # }
            # res.append(m2e)
            for cand in cands:
                ent_name = "ENTITY/"+cand[0].replace(" ", "_")
                entity2score[ent_name] = max(entity2score[cand[0]], cand[1])
        entities = list(entity2score.keys())
        entities, entity_ids, entity_embs = self.lookup_list(
            entities, "embeddings")
        entity_scores = [entity2score[ent] for ent in entities]
        return entities, entity_ids, entity_embs, entity_scores
