from torch.utils.data import Dataset
import torch
from datasets import load_dataset

from lsr.utils.dataset_utils import (
    read_collection,
    read_queries)


def iterable_train_triplets(triplet_part, queries_path, collection_path):
    # "data/msmarco_triplets.tsv"
    queries_dict = dict(read_queries(queries_path))
    docs_dict = read_collection(collection_path)

    def add_text(inp):
        qid = str(inp["qid"])
        q_text = queries_dict[qid]
        pid = str(inp["pid"])
        p_text = docs_dict[pid]
        nid = str(inp["nid"])
        n_text = docs_dict[nid]
        doc_text = [p_text, n_text]
        score = [inp["pscore"], inp["nscore"]]
        inp["query_id"] = qid
        inp["doc_id"] = [pid, nid]
        inp["query_text"] = q_text
        inp["doc_text"] = doc_text
        inp["score"] = score
        return inp
    data = load_dataset(
        "csv", data_files=triplet_part, delimiter="\t", names=["qid", "pid", "pscore", "nid", "nscore"])['train']
    data = data.to_iterable_dataset(num_shards=120)
    data = data.map(add_text)
    # data.set_format(columns=[
    #     'query_text', 'doc_text', 'score'])
    # data = data.with_format("torch")
    # assert isinstance(data, torch.utils.data.IterableDataset)
    return data
