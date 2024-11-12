from torch.utils.data import Dataset
import random
from lsr.utils.dataset_utils import (
    read_collection,
    read_queries,
    read_qrels,
    read_ce_score,
    read_entity_annotations,
    read_entity_embeddings
)


class TextCollection(Dataset):
    def __init__(self, ids, texts, entity_ids=None, entity_embs=None, entity_scores=None, tag="query") -> None:
        super().__init__()
        self.ids = ids
        self.texts = texts
        self.entity_ids = entity_ids
        self.entity_embs = entity_embs
        self.entity_scores = entity_scores
        self.id_key = f"{tag}_id"
        self.text_key = f"{tag}_text"
        self.ent_id_key = f"{tag}_entity_ids"
        self.ent_emb_key = f"{tag}_entity_embeddings"
        self.ent_score_key = f"{tag}_entity_probs"

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        if self.entity_ids and self.entity_embs and self.entity_scores:
            return {self.id_key: self.ids[index],
                    self.text_key: self.texts[index],
                    self.ent_id_key: self.entity_ids[index],
                    self.ent_emb_key: self.entity_embs[index],
                    self.ent_score_key: self.entity_scores[index]}
        else:
            return {self.id_key: self.ids[index], self.text_key: self.texts[index]}


class PredictionDataset:
    def __init__(self, qrels_path, queries_path, docs_path, num_documents=-1, query_field=["text"], doc_field=["text"], queries_ent_annotation_path=None, docs_ent_annotation_path=None, ent_embeddings_path=None) -> None:
        full_docs = read_collection(docs_path, text_fields=doc_field)
        full_queries = read_queries(queries_path, text_fields=query_field)
        self.qrels = read_qrels(qrels_path)
        if num_documents < 0:
            doc_ids = list(full_docs.keys())
            doc_texts = list(full_docs.values())
            query_ids, query_texts = list(zip(*full_queries))
        else:
            # keep queries and documents in qrels only
            # filter queries
            query_ids = list(self.qrels.keys())
            full_queries = dict(full_queries)
            query_texts = [full_queries[qid] for qid in query_ids]
            # filter documents
            doc_ids = set()
            for qid in self.qrels:
                doc_ids.update(list(self.qrels[qid].keys()))
            doc_ids = list(doc_ids)
            # if len(doc_ids) < num_documents:
            #     random.seed(42)
            #     all_ids = sorted(
            #         list(set(full_docs.keys()).difference(doc_ids)))
            #     num_sample = num_documents - len(doc_ids)
            #     random.shuffle(all_ids)
            #     sample_ids = all_ids[:num_sample]
            #     doc_ids = list(doc_ids) + sample_ids
            doc_texts = [full_docs[did] for did in doc_ids]
            print(f"Number of documents used for evaluation: {len(doc_texts)}")
        query_entity_ids = []
        query_entity_embs = []
        query_entity_scores = []
        doc_entity_ids = []
        doc_entity_embs = []
        doc_entity_scores = []
        if queries_ent_annotation_path and docs_ent_annotation_path and ent_embeddings_path:
            # Load entity annotatin
            entity2id, entity2emb = read_entity_embeddings(ent_embeddings_path)
            query_entities_annotation = read_entity_annotations(
                queries_ent_annotation_path, id_key="qid", link_key="links", filter=lambda x: not x in entity2id)
            doc_entities_annotation = read_entity_annotations(
                docs_ent_annotation_path, id_key="pid", link_key="passage", filter=lambda x: not x in entity2id)
            for qid in query_ids:
                per_query_entity_names, per_query_entity_scores = query_entities_annotation[
                    qid]
                per_query_entity_ids = [entity2id[ent_name]
                                        for ent_name in per_query_entity_names]
                per_query_entity_embs = [entity2emb[ent_name]
                                         for ent_name in per_query_entity_names]
                query_entity_ids.append(per_query_entity_ids)
                query_entity_embs.append(per_query_entity_embs)
                query_entity_scores.append(per_query_entity_scores)
            for did in doc_ids:
                per_doc_entity_names, per_doc_entity_scores = doc_entities_annotation[did]
                per_doc_entity_ids = [entity2id[ent_name]
                                      for ent_name in per_doc_entity_names]
                per_doc_entity_embs = [entity2emb[ent_name]
                                       for ent_name in per_doc_entity_names]
                doc_entity_ids.append(per_doc_entity_ids)
                doc_entity_embs.append(per_doc_entity_embs)
                doc_entity_scores.append(per_doc_entity_scores)
        self.docs = TextCollection(
            doc_ids, doc_texts, entity_ids=doc_entity_ids, entity_embs=doc_entity_embs, entity_scores=doc_entity_scores, tag="doc")
        self.queries = TextCollection(
            query_ids, query_texts, entity_ids=query_entity_ids, entity_embs=query_entity_embs, entity_scores=query_entity_scores, tag="query")
        # for qid in self.qrels:
        #     self.qrels[qid] = {did: 1 for did in self.qrels[qid]}
