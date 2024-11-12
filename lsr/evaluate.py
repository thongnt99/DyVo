from torch.cuda.amp import autocast
import argparse
from lsr.datasets.prediction_dataset import PredictionDataset
from lsr.datasets.text_entity_collator import DynamicDataCollator
from lsr.datasets.data_collator import DataCollator
from lsr.entity.precomputed_candidates import PrecomputedCandidate
from torch.utils.data import DataLoader
from lsr.models import DualSparseEncoder, DualSparseConfig
from pathlib import Path
import ir_measures
from pprint import pprint
from lsr.tokenizer import HFTokenizer
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from hydra.core.hydra_config import HydraConfig
from pprint import pprint
import logging
import wandb
import os
from datetime import datetime
import shutil
from tqdm import tqdm
import torch
import math
import subprocess
from ir_measures import *
from transformers import AutoTokenizer
import json
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser("")
parser.add_argument("--emlm", action="store_true", default=False)
parser.add_argument("--model_path", type=str,
                    default="outputs/qmlp_dmlm_emlm_msmarco_distil_l1_0.0_0.0001/10-17-2023/model")
parser.add_argument("--q_ent", type=str,
                    default="/projects/0/guse0488/dataset/entities/msmarco-passage/msmarco_queries.jsonl")
parser.add_argument("--d_ent", type=str,
                    default="/projects/0/guse0488/dataset/entities/msmarco-passage/msmarco_passages.jsonl")
parser.add_argument("--dataset", type=str, default="msmarco")


def to_cuda(inp_data):
    for key in inp_data:
        if isinstance(inp_data[key], torch.Tensor):
            inp_data[key] = inp_data[key].to("cuda")
    return inp_data


def evaluation_loop(model, query_dataloader, doc_dataloader, qrels, output_dir):
    eval_dir = Path(output_dir)/"tmp_eval"
    if not eval_dir.is_dir():
        eval_dir.mkdir(parents=True)
    queries_path = eval_dir/"queries.tsv"
    docs_dir = eval_dir/"docs"
    shutil.rmtree(docs_dir, ignore_errors=True)
    docs_dir.mkdir()
    index_dir = eval_dir/"index"
    if not index_dir.is_dir():
        index_dir.mkdir()
    run_path = eval_dir/"eval_run.trec"
    with open(queries_path, "w") as fquery:
        for batch_queries in tqdm(query_dataloader, desc=f"Encoding queries and saving to {queries_path}"):
            query_ids = batch_queries["query_ids"]
            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
                query_tok_ent_ids, query_tok_ent_weights = model.encode_queries(
                    **to_cuda(batch_queries["queries"]), to_dense=False)
                query_tok_ent_ids = query_tok_ent_ids.to("cpu").tolist()
                query_tok_ent_weights = torch.ceil(
                    query_tok_ent_weights*100).int().to("cpu").tolist()
            for qid, tokid_list, tokweight_list in zip(query_ids, query_tok_ent_ids, query_tok_ent_weights):
                toks = []
                for tokid, tokweight in zip(tokid_list, tokweight_list):
                    toks.extend([str(tokid)]*tokweight)
                toks_str = " ".join(toks)
                fquery.write(f"{qid}\t{toks_str}\n")
    num_partition = 60

    # fdoc = open(docs_path, "w")
    iter_per_patition = math.ceil(len(doc_dataloader)/num_partition)
    doc_iter = iter(doc_dataloader)
    for part_idx in tqdm(list(range(num_partition)), desc="Encoding documents"):
        docs_path = docs_dir/f"docs_{part_idx}.jsonl"
        with open(docs_path, "w") as fdoc:
            for _ in range(iter_per_patition):
                try:
                    batch_docs = next(doc_iter)
                except:
                    break
                doc_ids = batch_docs["doc_ids"]
                with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
                    doc_tok_ent_ids, doc_tok_ent_weights = model.encode_docs(
                        **to_cuda(batch_docs["doc_groups"]), to_dense=False)
                    doc_tok_ent_ids = doc_tok_ent_ids.to("cpu").tolist()
                    doc_tok_ent_weights = (
                        doc_tok_ent_weights*100).int().to("cpu").tolist()
                for doc_id, tokid_list, tokweight_list in zip(doc_ids, doc_tok_ent_ids, doc_tok_ent_weights):
                    tokid_list = [str(tokid) for tokid in tokid_list]
                    vector = {tokid: tokweight for tokid,
                              tokweight in zip(tokid_list, tokweight_list) if tokweight > 0}
                    doc_json = {"id": doc_id, "vector": vector}
                    fdoc.write(json.dumps(doc_json)+"\n")
        logger.info(f"Docs part {part_idx} written to {docs_path}")
    logger.info("-> Perform indexing")
    ANSERINI_INDEX_COMMAND = f"""/projects/0/guse0488/anserini-lsr/target/appassembler/bin/IndexCollection
        -collection JsonSparseVectorCollection 
        -input {docs_dir}  
        -index {index_dir}  
        -generator SparseVectorDocumentGenerator 
        -threads 18 
        -impact 
        -pretokenized
    """
    process = subprocess.run(ANSERINI_INDEX_COMMAND.split(), check=True)
    ANSERINI_RETRIEVE_COMMAND = f"""/projects/0/guse0488/anserini-lsr/target/appassembler/bin/SearchCollection
        -index {index_dir}  
        -topics {queries_path} 
        -topicreader TsvString 
        -output {run_path}  
        -impact 
        -pretokenized 
        -hits 1000 
        -parallelism 18"""
    process = subprocess.run(ANSERINI_RETRIEVE_COMMAND.split(), check=True)
    del doc_tok_ent_ids
    del doc_tok_ent_weights
    del query_tok_ent_ids
    del query_tok_ent_weights
    torch.cuda.empty_cache()
    run = list(ir_measures.read_trec_run(str(run_path)))
    if qrels is None:
        metrics = {}
    else:
        metrics = ir_measures.calc_aggregate(
            [MAP, MRR@10, R@5, R@10, R@100, R@1000, NDCG@10, NDCG@20], qrels, run)
    return run, metrics


if __name__ == "__main__":
    args = parser.parse_args()
    tokenizer = HFTokenizer("distilbert-base-uncased")
    if args.emlm:
        from lsr.models.mlm_emlm import TransformerMLMConfig, TransformerMLMSparseEncoder
        from lsr.models.mlp_emlm import TransformerMLPConfig, TransformerMLPSparseEncoder
        model_dir = Path(args.model_path)
        # Path(
        # "outputs/qmlp_dmlm_emlm_msmarco_distil_l1_0.0_0.0001/10-17-2023/")
        # entity_candidate_retrieval = lsr.data
        ent_candidate_retriever = PrecomputedCandidate(
            query_ent_path=args.q_ent, doc_ent_path=args.d_ent)
        data_collator = DynamicDataCollator(
            tokenizer, q_max_length=512, d_max_length=512, candidate_retriever=ent_candidate_retriever)
    else:
        from lsr.models import TransformerMLMConfig, TransformerMLMSparseEncoder
        from lsr.models import TransformerMLPConfig, TransformerMLPSparseEncoder
        # model_dir = Path(
        #     "outputs/qmlp_dmlm_msmarco_hn_l1_0.0_0.0001/10-27-2023/model")
        model_dir = Path(args.model_path)
        # model_dir = Path(
        # "outputs/qmlp_dmlm_msmarco_distil_l1_0.0_0.0001/10-18-2023/model")
        data_collator = DataCollator(
            tokenizer, q_max_length=512, d_max_length=512)
    query_encoder = TransformerMLPSparseEncoder.from_pretrained(
        model_dir/"query_encoder/")
    doc_encoder = TransformerMLMSparseEncoder.from_pretrained(
        model_dir/"doc_encoder/")
    config = DualSparseConfig.from_pretrained(
        model_dir)
    model = DualSparseEncoder(
        query_encoder=query_encoder, doc_encoder=doc_encoder, config=config)
    model.to("cuda")
    if args.dataset == "robust04":
        dataset = PredictionDataset(docs_path='irds:disks45/nocr/trec-robust-2004',
                                    queries_path='irds:disks45/nocr/trec-robust-2004', doc_field=["title", "body"], query_field=["description"], qrels_path='irds:disks45/nocr/trec-robust-2004')
    elif args.dataset == "msmarco":
        dataset = PredictionDataset(docs_path='irds:msmarco-passage',
                                    queries_path='irds:msmarco-passage/dev/small', query_field=["text"], qrels_path='irds:msmarco-passage/dev/small')
    elif args.dataset == "codec":
        dataset = PredictionDataset(docs_path='irds:codec', query_field=['query'],
                                    queries_path='irds:codec', qrels_path='irds:codec')
    elif args.dataset == "core18":
        dataset = PredictionDataset(docs_path='irds:wapo/v2/trec-core-2018',
                                    queries_path='irds:wapo/v2/trec-core-2018', doc_field=["title", "body"], query_field=["description"], qrels_path='irds:wapo/v2/trec-core-2018')
    elif args.dataset == "news18":
        dataset = PredictionDataset(docs_path='irds:wapo/v2/trec-news-2018',
                                    queries_path='irds:wapo/v2/trec-news-2018', doc_field=["title", "body"], qrels_path='irds:wapo/v2/trec-news-2018')
    elif args.dataset == "news19":
        dataset = PredictionDataset(docs_path='irds:wapo/v2/trec-news-2019',
                                    queries_path='irds:wapo/v2/trec-news-2019', doc_field=["title", "body"], qrels_path='irds:wapo/v2/trec-news-2019')
    query_data_loader = DataLoader(
        dataset.queries, num_workers=18, shuffle=False, collate_fn=data_collator, batch_size=128)
    doc_data_loader = DataLoader(
        dataset.docs, num_workers=18, shuffle=False, collate_fn=data_collator, batch_size=64)
    test_run, metrics = evaluation_loop(model, query_data_loader, doc_data_loader, dataset.qrels,
                                        model_dir/args.dataset)
    result_path = model_dir / args.dataset / "test_result.json"
    print(metrics)
    metrics = {str(m): v for m, v in metrics.items()}
    json.dump(metrics, open(result_path, "w"))
