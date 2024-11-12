import torch
from tqdm import tqdm


class DataCollator:
    "Tokenize and batch of (query, pos, neg, pos_score, neg_score)"

    def __init__(self, tokenizer, q_max_length, d_max_length):
        self.tokenizer = tokenizer
        self.q_max_length = q_max_length
        self.d_max_length = d_max_length

    def __call__(self, batch):
        batch_queries = []
        batch_query_ids = []
        batch_query_entity_ids = []
        batch_query_entity_embs = []
        batch_query_entity_scores = []
        batch_query_entity_mask = []

        batch_doc_entity_ids = []
        batch_doc_ids = []
        batch_doc_entity_embs = []
        batch_doc_entity_scores = []
        batch_doc_entity_mask = []
        batch_docs = []
        batch_scores = []
        for example in batch:
            # collate queries
            if "query_id" in example:
                batch_query_ids.append(example["query_id"])
            if "query_text" in example:
                batch_queries.append(example["query_text"])
                batch_query_entity_ids.append(example["query_entity_ids"])
                batch_query_entity_scores.append(
                    example["query_entity_scores"])
                batch_query_entity_embs.append(
                    example["query_entity_embs"])
            # collate docs
            if "doc_id" in example:
                if isinstance(example["doc_id"], list):
                    batch_doc_ids.extend(example["doc_id"])
                else:
                    batch_doc_ids.append(example["doc_id"])
            if "doc_text" in example:
                if isinstance(example["doc_text"], list):
                    batch_docs.extend(example["doc_text"])
                    batch_doc_entity_ids.extend(example["doc_entity_ids"])
                    batch_doc_entity_scores.extend(
                        example["doc_entity_scores"])
                    batch_doc_entity_embs.extend(
                        example["doc_entity_embs"])
                else:
                    batch_docs.append(example["doc_text"])
                    batch_doc_entity_ids.append(example["doc_entity_ids"])
                    batch_doc_entity_scores.append(
                        example["doc_entity_scores"])
                    batch_doc_entity_embs.append(
                        example["doc_entity_embs"])
            if "score" in example:
                batch_scores.append(example["score"])

        # padding query enties
        if len(batch_queries) > 0:
            max_ent_q = max(1, max([len(ent_list)
                            for ent_list in batch_query_entity_ids]))
            for idx in range(len(batch_queries)):
                num_pad = max_ent_q - len(batch_query_entity_ids[idx])
                batch_query_entity_ids[idx] += [0] * num_pad
                batch_query_entity_scores[idx] += [0.0] * num_pad
                batch_query_entity_embs[idx] += [
                    torch.zeros(300).tolist()]*num_pad
                batch_query_entity_mask.append(
                    [1]*(max_ent_q-num_pad) + [0]*num_pad)
            batch_query_entity_ids = torch.tensor(
                batch_query_entity_ids) + self.tokenizer.tokenizer.vocab_size
            batch_query_entity_scores = torch.tensor(batch_query_entity_scores)
            batch_query_entity_embs = torch.tensor(batch_query_entity_embs)
            batch_query_entity_mask = torch.tensor(batch_query_entity_mask)
            tokenized_queries = dict(self.tokenizer(
                batch_queries,
                padding=True,
                truncation=True,
                max_length=self.q_max_length,
                return_tensors="pt",
                return_special_tokens_mask=True,
            ))
            tokenized_queries.update(
                {"entity_ids": batch_query_entity_ids, "entity_scores": batch_query_entity_scores, "entity_embs": batch_query_entity_embs})

        else:
            tokenized_queries = {}
        # padding doc enties
        if len(batch_docs) > 0:
            max_ent_d = max(1, max([len(ent_list)
                            for ent_list in batch_doc_entity_ids]))
            for idx in range(len(batch_docs)):
                num_pad = max_ent_d - len(batch_doc_entity_ids[idx])
                batch_doc_entity_ids[idx] = batch_doc_entity_ids[idx] + \
                    [0] * num_pad
                batch_doc_entity_scores[idx] = batch_doc_entity_scores[idx] + \
                    [0.0] * num_pad
                batch_doc_entity_embs[idx] = batch_doc_entity_embs[idx] + \
                    [torch.zeros(300).tolist()]*num_pad
                batch_doc_entity_mask.append(
                    [1]*(max_ent_d-num_pad) + [0]*num_pad)

            batch_doc_entity_ids = torch.tensor(
                batch_doc_entity_ids) + self.tokenizer.tokenizer.vocab_size
            batch_doc_entity_scores = torch.tensor(batch_doc_entity_scores)
            batch_doc_entity_embs = torch.tensor(batch_doc_entity_embs)
            batch_doc_entity_mask = torch.tensor(batch_doc_entity_mask)
            tokenized_docs = dict(self.tokenizer(
                batch_docs,
                padding=True,
                truncation=True,
                max_length=self.d_max_length,
                return_tensors="pt",
                return_special_tokens_mask=True,
            ))
            tokenized_docs.update(
                {"entity_ids": batch_doc_entity_ids, "entity_scores": batch_doc_entity_scores, "entity_embs": batch_doc_entity_embs})

        else:
            tokenized_docs = {}
        return {
            "query_ids": batch_query_ids,
            "queries": tokenized_queries,
            "doc_ids": batch_doc_ids,
            "docs_batch": tokenized_docs,
            "labels": torch.tensor(batch_scores) if len(batch_scores) > 0 else None,
        }


class DynamicDataCollator:
    "Dynamic DataCollator that retrieves the entity candidates on the fly."

    def __init__(self, tokenizer, q_max_length, d_max_length, candidate_retriever, entity_dim=300):
        self.tokenizer = tokenizer
        self.q_max_length = q_max_length
        self.d_max_length = d_max_length
        self.candidate_retriever = candidate_retriever
        self.entity_dim = entity_dim

    def pad_one_group(self, text_group, ent_id_group, ent_mask_group):
        max_ent_d = max(1, max([len(ent_list)
                                for ent_list in ent_id_group]))
        batch_size = len(text_group)
        ent_id_group = list(ent_id_group)
        ent_mask_group = list(ent_mask_group)
        for idx in range(batch_size):
            num_pad = max_ent_d - len(ent_id_group[idx])
            ent_id_group[idx] = ent_id_group[idx] + \
                [0] * num_pad
            ent_mask_group[idx] = ent_mask_group[idx] + \
                [0] * num_pad
        ent_id_group = torch.tensor(
            ent_id_group) + self.tokenizer.tokenizer.vocab_size
        ent_mask_group = torch.tensor(ent_mask_group)
        inp_group = dict(self.tokenizer(
            text_group,
            padding=True,
            truncation=True,
            max_length=self.d_max_length,
            return_tensors="pt",
            return_special_tokens_mask=True,
        ))
        inp_group.update(
            {"entity_ids": ent_id_group, "entity_masks": ent_mask_group})
        return inp_group

    def pad_multi_groups(self, batch_texts, batch_entity_ids, batch_entity_masks):
        text_groups = list(zip(*batch_texts))
        ent_groups = list(zip(*batch_entity_ids))
        mask_groups = list(zip(*batch_entity_masks))
        multi_groups = []
        for text_group, ent_id_group,  ent_mask_group in zip(text_groups, ent_groups,  mask_groups):
            inp_group = self.pad_one_group(
                text_group, ent_id_group,  ent_mask_group)
            multi_groups.append(inp_group)
        return multi_groups

    def retrieve_entities(self, texts, text_ids, type):
        batch_candidates = [self.candidate_retriever.retrieve_candidates(
            text, text_id, type=type) for text, text_id in zip(texts, text_ids)]
        entity_ids, entity_masks = list(
            zip(*batch_candidates))
        return entity_ids, entity_masks

    def __call__(self, batch):
        batch_queries = []
        batch_query_ids = []
        batch_query_entity_ids = []
        batch_query_entity_mask = []

        batch_doc_entity_ids = []
        batch_doc_ids = []
        batch_doc_entity_mask = []
        batch_docs = []
        batch_scores = []
        for example in batch:
            # collate queries
            if "query_id" in example:
                batch_query_ids.append(example["query_id"])
            if "query_text" in example:
                batch_queries.append(example["query_text"])
                query_entity_ids, query_entity_mask = self.candidate_retriever.retrieve_candidates(
                    example["query_text"], example["query_id"], type="query")
                batch_query_entity_ids.append(query_entity_ids)
                batch_query_entity_mask.append(query_entity_mask)
            # collate docs
            if "doc_id" in example:
                batch_doc_ids.append(example["doc_id"])
            if "doc_text" in example:
                batch_docs.append(example["doc_text"])
                if isinstance(example["doc_text"], str):
                    doc_entity_ids, doc_entity_mask = self.candidate_retriever.retrieve_candidates(
                        example["doc_text"], example["doc_id"], type="doc")
                else:
                    doc_entity_ids, doc_entity_mask = self.retrieve_entities(
                        example["doc_text"], example["doc_id"], type="doc")
                batch_doc_entity_ids.append(doc_entity_ids)
                batch_doc_entity_mask.append(doc_entity_mask)
            if "score" in example:
                batch_scores.append(example["score"])

        # padding query enties
        if len(batch_queries) > 0:
            query_groups = self.pad_one_group(
                batch_queries, batch_query_entity_ids, batch_query_entity_mask)
        else:
            query_groups = {}
        # padding doc enties
        if len(batch_docs) > 0:
            if isinstance(batch_docs[0], str):
                doc_groups = self.pad_one_group(
                    batch_docs, batch_doc_entity_ids, batch_doc_entity_mask)
            else:
                doc_groups = self.pad_multi_groups(
                    batch_docs, batch_doc_entity_ids, batch_doc_entity_mask)
        else:
            doc_groups = {}

        return {
            "queries": query_groups,
            "doc_groups": doc_groups,
            "labels": torch.tensor(batch_scores) if len(batch_scores) > 0 else None,
            "query_ids": batch_query_ids,
            "doc_ids": batch_doc_ids,
        }
