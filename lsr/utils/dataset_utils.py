from tqdm import tqdm
import ir_datasets
# import irds_robust_anserini
import json
from datasets import DownloadManager
from collections import defaultdict
import gzip
import pickle
from pathlib import Path
import requests
import json
import sys
from datasets import load_dataset
import numpy as np
import pandas as pd
import glob
import torch
import hashlib
import copy
IRDS_PREFIX = "irds:"
HFG_PREFIX = "hfds:"

in_mem_cache = {}


def read_entity_embeddings(emb_path, refresh=False):
    cache_key = hashlib.md5(
        ("read_entity_embeddings::"+emb_path).encode()).hexdigest()
    print(f"Loading entity embeddings from {emb_path}")
    if cache_key in in_mem_cache:
        print("Previosly loaded. Read from memory cache.")
        return in_mem_cache[cache_key]
    df = pd.read_parquet(emb_path)
    entity_names = df["names"]
    entity_embs = df["embs"].array
    entity_embs = [emb.tolist() for emb in entity_embs]
    entity2id = dict(zip(entity_names, range(len(entity_names))))
    entity2emb = dict(zip(entity_names, entity_embs))
    in_mem_cache[cache_key] = (entity2id, entity2emb)
    return entity2id, entity2emb


def read_entity_annotations(path, id_key, link_key, filter=lambda x: False):
    cache_key = hashlib.md5(
        (f"read_entity_annotations::{path}::id_key::{id_key}::link_key::{link_key}").encode()).hexdigest()
    if cache_key in in_mem_cache:
        print(f"Previosly read. Load annotation: {path} from memory cache")
        return in_mem_cache[cache_key]
    else:
        res = defaultdict(lambda: ([], []))
        with open(path, "r") as f:
            for line in tqdm(f, desc=f"Loading entity annotation from {path}"):
                ann = json.loads(line.strip())
                text_id = ann[id_key]
                entity2score = {}
                for ent in ann[link_key]:
                    if filter(ent["entity"]):
                        continue
                    score = ent["details"]["md_score"]
                    if ent["entity"] in entity2score:
                        entity2score[ent["entity"]] = max(
                            entity2score[ent["entity"]], score)
                    else:
                        entity2score[ent["entity"]] = score
                res[str(text_id)] = (list(entity2score.keys()),
                                     list(entity2score.values()))
        in_mem_cache[cache_key] = res
        return res


def read_collection(collection_path: str, text_fields=["text"]):
    cache_key = hashlib.md5(
        ("read_collection::"+collection_path).encode()).hexdigest()
    if cache_key in in_mem_cache:
        print(f"Previosly read. Load {collection_path} from memory cache")
        return in_mem_cache[cache_key]
    doc_dict = {}
    if collection_path.startswith(IRDS_PREFIX):
        irds_name = collection_path.replace(IRDS_PREFIX, "")
        dataset = ir_datasets.load(irds_name)
        for doc in tqdm(
            dataset.docs_iter(),
            desc=f"Loading doc collection from ir_datasets: {irds_name}",
        ):
            doc_id = doc.doc_id
            texts = [getattr(doc, field) for field in text_fields]
            texts = [text for text in texts if text is not None]
            text = " ".join(texts)
            doc_dict[doc_id] = text
    elif collection_path.startswith(HFG_PREFIX):
        hfg_name = collection_path.replace(HFG_PREFIX, "")
        dataset = load_dataset(hfg_name)
        for row in tqdm(
            dataset["passage"],
            desc=f"Loading data from HuggingFace datasets: {hfg_name}",
        ):
            doc_dict[row["id"]] = row["text"]
    else:
        with open(collection_path, "r") as f:
            for line in tqdm(f, desc=f"Reading doc collection from {collection_path}"):
                doc_id, doc_text = line.strip().split("\t")
                doc_dict[doc_id] = doc_text
    in_mem_cache[cache_key] = doc_dict
    return doc_dict


def read_queries(queries_path: str, text_fields=["text"]):
    queries = []
    if queries_path.startswith(IRDS_PREFIX):
        irds_name = queries_path.replace(IRDS_PREFIX, "")
        dataset = ir_datasets.load(irds_name)
        for query in tqdm(
            dataset.queries_iter(),
            desc=f"Loading queries from ir_datasets: {queries_path}",
        ):
            query_id = query.query_id
            if "wapo/v2/trec-news" in irds_name:
                doc_id = query.doc_id
                doc = dataset.docs_store().get(doc_id)
                text = doc.title + " " + doc.body
            else:
                texts = [getattr(query, field) for field in text_fields]
                text = " ".join(texts)
            queries.append([query_id, text])
    else:
        with open(queries_path, "r") as f:
            for line in tqdm(f, desc=f"Reading queries from {queries_path}"):
                query_id, query_text = line.strip().split("\t")
                queries.append([query_id, query_text])
    return queries


def read_qrels(qrels_path: str, rel_threshold=0):
    qid2pos = defaultdict(dict)
    if qrels_path.startswith(IRDS_PREFIX):
        irds_name = qrels_path.replace(IRDS_PREFIX, "")
        dataset = ir_datasets.load(irds_name)
        for qrel in dataset.qrels_iter():
            # if qrel.relevance > rel_threshold:
            qid, did = qrel.query_id, qrel.doc_id
            qid2pos[qid][did] = qrel.relevance
    else:
        qrels = json.load(open(qrels_path, "r"))
        for qid in qrels:
            qid2pos[str(qid)] = {str(did): qrels[qid][did]
                                 for did in qrels[qid]}
    return qid2pos


file_map = {
    "sentence-transformers/msmarco-hard-negatives": "https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/cross-encoder-ms-marco-MiniLM-L-6-v2-scores.pkl.gz"
}


def read_ce_score(ce_path: str):
    if ce_path.endswith(".json"):
        return json.load(open(ce_path, "r"))
    if ce_path.startswith(HFG_PREFIX):
        hf_name = ce_path.replace(HFG_PREFIX, "")
        _url = file_map[hf_name]
        dl_manager = DownloadManager()
        ce_path = dl_manager.download(_url)
    res = {}
    with gzip.open(ce_path, "rb") as f:
        data = pickle.load(f)
        for qid in tqdm(data, desc=f"Preprocessing data from {ce_path}"):
            res[str(qid)] = {str(did): data[qid][did] for did in data[qid]}
    return res


def read_triplets(triplet_path: str):
    triplets = []
    query2pos = defaultdict(list)
    query2neg = defaultdict(list)
    if triplet_path.startswith(IRDS_PREFIX):
        irds_name = triplet_path.replace(IRDS_PREFIX, "")
        dataset = ir_datasets.load(irds_name)
        for docpair in tqdm(
            dataset.docpairs_iter(
            ), f"Loading triplets from ir_datasets: {irds_name}"
        ):
            qid, pos_id, neg_id = docpair.query_id, docpair.doc_id_a, docpair.doc_id_b
            triplets.append((qid, pos_id, neg_id))
            query2pos[qid].append(pos_id)
            query2neg[qid].append(neg_id)
    else:
        with open(triplet_path) as f:
            for line in tqdm(f, desc=f"Reading triplets from {triplet_path}"):
                qid, pos_id, neg_id = line.strip().split("\t")
                triplets.append((qid, pos_id, neg_id))
                query2pos[qid].append(pos_id)
                query2neg[qid].append(neg_id)
    return triplets, query2pos, query2neg


def download_file(url, file_path):
    """
    Downloads file and store in a path (Code borrow from Sentence Transformers)
    Arguments
    ---------
    url: str
        link of the file
    path:    
        path to store the file
    """
    file_path = Path(file_path)
    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True, exist_ok=True)
    req = requests.get(url, stream=True)
    if req.status_code != 200:
        print(
            f"Error when trying to download {url}. Response {req.status_code}",
            file=sys.stderr,
        )
        req.raise_for_status()
        return
    with open(file_path, "wb") as f:
        content_length = req.headers.get("Content-Length")
        total = int(content_length) if content_length is not None else None
        progress = tqdm(unit="B", total=total, unit_scale=True)
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:
                progress.update(len(chunk))
                f.write(chunk)
    progress.close()
