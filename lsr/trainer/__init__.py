import torch
import os
from typing import Dict, List, Optional
from torch.utils.data import DataLoader, Dataset
import transformers
import logging
from lsr.models import DualSparseEncoder
from collections import defaultdict, OrderedDict
import time
import ir_measures
from ir_measures import *
import subprocess
from transformers.trainer_utils import EvalLoopOutput, speed_metrics
from tqdm import tqdm
from transformers.trainer_utils import PredictionOutput
import json
import shutil
import math
import os
from pathlib import Path
from heapq import heappop, heappush
logger = logging.getLogger(__name__)
LOSS_NAME = "loss.pt"

USER = os.environ.get('USER')


def make_run_file(score_dict, output_file):
    run = defaultdict(dict)
    with open(output_file, "w") as f:
        for query_id in tqdm(score_dict, desc=f"Run file saving to: {output_file}"):
            pairs = sorted(score_dict[query_id],
                           key=lambda x: x[0], reverse=True)
            for rank, pair in enumerate(pairs):
                doc_score, doc_id = pair
                run[query_id][doc_id] = doc_score
                f.write(f"{query_id}\t0\t{doc_id}\t{rank}\t{doc_score}\tLSR\n")
    return run

def update_topk(score_dict, qid, did, score, topk):
    if len(score_dict[qid]) < topk:
        heappush(score_dict[qid], (score, did))
    else:
        if score > score_dict[qid][0][0]:
            heappop(score_dict[qid])
            heappush(score_dict[qid], (score, did))

def write_reprsentation_to_file(output_file, batch_ids, batch_tok_ids, batch_tok_weights):
    with open(output_file, "a") as f:
        for ith in range(len(batch_ids)):
            text_id = batch_ids[ith]
            list_tok_ids = batch_tok_ids[ith].to("cpu").tolist()
            list_tok_weights = batch_tok_weights[ith].to("cpu").tolist()
            vector = {str(tid): tw for tid, tw in zip(
                list_tok_ids, list_tok_weights) if tw > 0}
            row_obj = {"id": text_id, "vector": vector}
            f.write(json.dumps(row_obj)+"\n")
            
class HFTrainer(transformers.trainer.Trainer):
    """Customized Trainer from Huggingface's Trainer"""

    def __init__(self, *args, loss=None, inverted_index=False, **kwargs) -> None:
        super(HFTrainer, self).__init__(*args, **kwargs)
        self.loss = loss
        self.customed_log = defaultdict(lambda: 0.0)
        self.tokenizer = self.data_collator.tokenizer
        self.inverted_index = inverted_index

    def _maybe_log_save_evaluate(
        self, tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval
    ):
        if self.control.should_log:
            log = {}
            for metric in self.customed_log:
                log[metric] = (
                    self._nested_gather(
                        self.customed_log[metric]).mean().item()
                )
                log[metric] = round(
                    (
                        log[metric]
                        / (self.state.global_step - self._globalstep_last_logged)
                        / self.args.gradient_accumulation_steps
                    ),
                    4,
                )
            self.log(log)
            for metric in self.customed_log:
                self.customed_log[metric] -= self.customed_log[metric]
            self.control.should_log = True
        super()._maybe_log_save_evaluate(
            tr_loss, grad_norm, model, trial, epoch, ignore_keys_for_eval
        )

    def _load_optimizer_and_scheduler(self, checkpoint):
        super()._load_optimizer_and_scheduler(checkpoint)
        if checkpoint is None:
            return
        if os.path.join(checkpoint, LOSS_NAME):
            self.loss.load_state_dict(torch.load(
                os.path.join(checkpoint, LOSS_NAME)))

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute loss
        """
        loss_output, q_reg, d_reg, to_log = model(self.loss, **inputs)
        for log_metric in to_log:
            self.customed_log[log_metric] += to_log[log_metric]
        return loss_output + q_reg + d_reg

    def save_model(self, model_dir=None, _internal_call=False):
        """Save model checkpoint"""
        logger.info("Saving model checkpoint to %s", model_dir)
        if model_dir is None:
            model_dir = os.path.join(self.args.output_dir, "model")
        self.model.save_pretrained(model_dir)
        if self.tokenizer is not None:
            tokenizer_path = os.path.join(model_dir, "tokenizer")
            self.tokenizer.save_pretrained(tokenizer_path)
        loss_path = os.path.join(model_dir, LOSS_NAME)
        logger.info("Saving loss' state to %s", loss_path)
        torch.save(self.loss.state_dict(), loss_path)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        """Load from a checkpoint to continue traning"""
        # Load model from checkpoint
        logger.info("Loading model's weight from %s", resume_from_checkpoint)
        self.model.load_state_dict(
            self.model.from_pretrained(
                resume_from_checkpoint).state_dict()
        )

    def _load_best_model(self):
        logger.info(
            f"Loading best model from {self.state.best_model_checkpoint} (score: {self.state.best_metric}).")
        self._load_from_checkpoint(self.state.best_model_checkpoint)

    def evaluation_loop_iterative_vectorized(self, query_dataloader, doc_dataloader, qrels=None, run_path=None, topk=10000, save_representation=False):
        eval_dir = Path(self.args.output_dir)/"tmp_eval"
        emb_dir = Path(f"/scratch-shared/{USER}/lsr-entities") / eval_dir
        queries_path = emb_dir/"queries.jsonl"
        docs_dir = emb_dir/"docs"
        try:
            os.remove(queries_path)
            shutil.rmtree(docs_dir, ignore_errors=True)
        except:
            pass
        if not eval_dir.is_dir():
            eval_dir.mkdir(exist_ok=True, parents=True)
        if not emb_dir.is_dir():
            emb_dir.mkdir(exist_ok=True, parents=True)
        docs_dir.mkdir()
        if not run_path:
            run_path = eval_dir/"eval_run.trec"
        self.model.eval()
        query_ids = []
        query_tok_ids = []
        query_tok_weights = []
        for batch_queries in tqdm(query_dataloader, desc=f"Encoding queries. reps saved to {queries_path}"):
            batch_query_ids = batch_queries["query_ids"]
            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
                batch_query_tok_ent_ids, batch_query_tok_ent_weights, _ = self.model.encode_queries(
                    **self._prepare_inputs(batch_queries["queries"]), to_dense=False)
                batch_query_tok_ent_ids = batch_query_tok_ent_ids
                batch_query_tok_ent_weights = batch_query_tok_ent_weights
            query_ids.extend(batch_query_ids)
            query_tok_ids.extend(list(batch_query_tok_ent_ids))
            query_tok_weights.extend(list(batch_query_tok_ent_weights))

        write_reprsentation_to_file(queries_path, query_ids, query_tok_ids, query_tok_weights)

        query2doc_scores = {qid: [] for qid in query_ids}
        partition_size = math.ceil(len(doc_dataloader)*1.0/60)
        for batch_idx, batch_docs in enumerate(tqdm(doc_dataloader, desc=f"Encoding & Scoring documents. reps saved to {docs_dir}")):
            batch_doc_ids = batch_docs["doc_ids"]
            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
                batch_doc_tok_ent_ids, batch_doc_tok_ent_weights, _ = self.model.encode_docs(
                    **self._prepare_inputs(batch_docs["doc_groups"]), to_dense=False)
            for ith_q in range(len(query_ids)):
                device = batch_doc_tok_ent_ids.device
                query_tok_ent_ids = query_tok_ids[ith_q].to(device)
                query_tok_ent_weights = query_tok_weights[ith_q].to(device)
                exact_match_mask = query_tok_ent_ids.unsqueeze(
                    0).unsqueeze(-1) == batch_doc_tok_ent_ids.unsqueeze(-2)
                interaction = query_tok_ent_weights.unsqueeze(
                    0).unsqueeze(-1) * batch_doc_tok_ent_weights.unsqueeze(-2)
                batch_scores = (exact_match_mask * interaction).sum(dim=-1).sum(dim=-1).tolist()
                for doc_id, score in zip(batch_doc_ids, batch_scores):
                    update_topk(query2doc_scores,
                                query_ids[ith_q], doc_id, score, topk)
            partition_idx = batch_idx//partition_size
            partition_path = docs_dir/f"part_{partition_idx}.jsonl"
            if save_representation or partition_idx == 0:
                write_reprsentation_to_file(
                    partition_path, batch_doc_ids, batch_doc_tok_ent_ids, batch_doc_tok_ent_weights)
        run = make_run_file(query2doc_scores, run_path)
        if qrels is None:
            metrics = {}
        else:
            metrics = ir_measures.calc_aggregate(
                [MRR@10, R@5, R@10, R@100, R@1000, R@100000, NDCG@10, NDCG@20, MAP], qrels, run)
        self.model.train()
        return run, metrics
    
    def evaluation_loop_iterative(self, query_dataloader, doc_dataloader, qrels=None, run_path=None):
        eval_dir = Path(self.args.output_dir)/"tmp_eval"
        emb_dir = Path(f"/scratch-shared/{USER}/lsr-entities")/eval_dir
        if not eval_dir.is_dir():
            eval_dir.mkdir()
            emb_dir.mkdir(parents=True, exist_ok=True)
        queries_path = emb_dir/"queries.jsonl"
        docs_path = emb_dir/"docs.jsonl"
        if not run_path:
            run_path = eval_dir/"eval_run.trec"
        try:
            os.remove(queries_path)
            os.remove(docs_path)
        except:
            pass
        self.model.eval()
        qid2rep = defaultdict(dict)
        with open(queries_path, "w") as fquery:
            for batch_queries in tqdm(query_dataloader, desc=f"Encoding queries and saving to {queries_path}"):
                batch_query_ids = batch_queries["query_ids"]
                with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
                    batch_query_tok_ent_ids, batch_query_tok_ent_weights = self.model.encode_queries(
                        **self._prepare_inputs(batch_queries["queries"]), to_dense=False)
                    batch_query_tok_ent_ids = batch_query_tok_ent_ids.to(
                        "cpu").tolist()
                    batch_query_tok_ent_weights = batch_query_tok_ent_weights.to(
                        "cpu").tolist()
                for qid, tokid_list, tokweight_list in zip(batch_query_ids, batch_query_tok_ent_ids, batch_query_tok_ent_weights):
                    w2w = {str(tok_id): tok_weight for tok_id,
                           tok_weight in zip(tokid_list, tokweight_list) if tok_weight > 0}
                    row = {"id":  qid, "vector": w2w}
                    row = json.dumps(row)
                    fquery.write(row+"\n")
                    qid2rep[qid] = w2w
        did2rep = defaultdict(dict)
        with open(docs_path, "w") as fdoc:
            for batch_docs in tqdm(doc_dataloader, desc=f"Encoding documents and saving to {docs_path}"):
                batch_doc_ids = batch_docs["doc_ids"]
                with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
                    batch_doc_tok_ent_ids, batch_doc_tok_ent_weights = self.model.encode_docs(
                        **self._prepare_inputs(batch_docs["doc_groups"]), to_dense=False)
                    batch_doc_tok_ent_ids = batch_doc_tok_ent_ids.to(
                        "cpu").tolist()
                    batch_doc_tok_ent_weights = batch_doc_tok_ent_weights.to(
                        "cpu").tolist()
                for doc_id, tokid_list, tokweight_list in zip(batch_doc_ids, batch_doc_tok_ent_ids, batch_doc_tok_ent_weights):
                    tokid_list = [str(tokid) for tokid in tokid_list]
                    vector = {tokid: tokweight for tokid,
                              tokweight in zip(tokid_list, tokweight_list) if tokweight > 0}
                    doc_json = {"id": doc_id, "vector": vector}
                    fdoc.write(json.dumps(doc_json)+"\n")
                    did2rep[doc_id] = vector
        run = defaultdict(dict)
        for q_id in tqdm(qid2rep, desc=f"Interatively computing query-doc scores -> {run_path}"):
            for d_id in did2rep:
                score = 0.0
                for tok in qid2rep[q_id]:
                    if tok in did2rep[d_id]:
                        score += qid2rep[q_id][tok]*did2rep[d_id][tok]
                if score > 0:
                    run[q_id][d_id] = score
        json.dump(run, open(run_path, "w"))
        if qrels is None:
            metrics = {}
        else:
            metrics = ir_measures.calc_aggregate(
                [MRR@10, R@5, R@10, R@100, R@1000, R@100000, NDCG@10, NDCG@20, MAP], qrels, run)
        self.model.train()
        return run, metrics

    def evaluation_loop(self, query_dataloader, doc_dataloader, qrels=None, run_path=None):
        eval_dir = Path(self.args.output_dir)/"tmp_eval"
        emb_dir = Path(f"/scratch-shared/{USER}/lsr-entities")/eval_dir
        queries_path = emb_dir/"queries.tsv"
        docs_dir = emb_dir/"docs"
        try:
            os.remove(queries_path)
            shutil.rmtree(docs_dir, ignore_errors=True)
        except:
            pass
        if not eval_dir.is_dir():
            eval_dir.mkdir(exist_ok=True, parents=True)
        if not emb_dir.is_dir():
            emb_dir.mkdir(exist_ok=True, parents=True)
        docs_dir.mkdir(exist_ok=True, parents=True)
        index_dir = eval_dir/"index"
        if not index_dir.is_dir():
            index_dir.mkdir()
        if not run_path:
            run_path = eval_dir/"eval_run.trec"
        self.model.eval()
        with open(queries_path, "w") as fquery:
            for batch_queries in tqdm(query_dataloader, desc=f"Encoding queries and saving to {queries_path}"):
                query_ids = batch_queries["query_ids"]
                with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
                    query_tok_ent_ids, query_tok_ent_weights, *_ = self.model.encode_queries(
                        **self._prepare_inputs(batch_queries["queries"]), to_dense=False)
                    query_tok_ent_ids = query_tok_ent_ids.to("cpu").tolist()
                    query_tok_ent_weights = torch.ceil(
                        query_tok_ent_weights*100).int().to("cpu").tolist()
                for qid, tokid_list, tokweight_list in zip(query_ids, query_tok_ent_ids, query_tok_ent_weights):
                    toks = []
                    for tokid, tokweight in zip(tokid_list, tokweight_list):
                        toks.extend([str(tokid)]*tokweight)
                    toks_str = " ".join(toks)
                    if toks_str:
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
                        doc_ids = batch_docs["doc_ids"]
                        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
                            doc_tok_ent_ids, doc_tok_ent_weights, *_ = self.model.encode_docs(
                                **self._prepare_inputs(batch_docs["doc_groups"]), to_dense=False)
                            doc_tok_ent_ids = doc_tok_ent_ids.to(
                                "cpu").tolist()
                            doc_tok_ent_weights = (
                                doc_tok_ent_weights*100).int().to("cpu").tolist()
                        for doc_id, tokid_list, tokweight_list in zip(doc_ids, doc_tok_ent_ids, doc_tok_ent_weights):
                            tokid_list = [str(tokid) for tokid in tokid_list]
                            vector = {tokid: tokweight for tokid,
                                      tokweight in zip(tokid_list, tokweight_list) if tokweight > 0}
                            doc_json = {"id": doc_id, "vector": vector}
                            fdoc.write(json.dumps(doc_json)+"\n")
                    except:
                        pass
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
                [MRR@10, R@5, R@10, R@100, R@1000, NDCG@10, NDCG@20, MAP], qrels, run)
        self.model.train()
        return run, metrics

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        """
        Run evaluation and returns metrics.

        The calling script will be responsible for providing a method to compute metrics, as they are task-dependent
        (pass it to the init `compute_metrics` argument).

        You can also subclass and override this method to inject custom behavior.

        Args:
            eval_dataset (`Dataset`, *optional*):
                Pass a dataset if you wish to override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns
                not accepted by the `model.forward()` method are automatically removed. It must implement the `__len__`
                method.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is "eval" (default)

        Returns:
            A dictionary containing the evaluation loss and the potential metrics computed from the predictions. The
            dictionary also contains the epoch number which comes from the training state.
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        doc_data_loader = self.get_eval_dataloader(
            eval_dataset.docs)
        query_dataloader = self.get_eval_dataloader(
            eval_dataset.queries)
        start_time = time.time()
        if self.inverted_index:
            run, metrics = self.evaluation_loop(
                query_dataloader, doc_data_loader, eval_dataset.qrels)
        else:
            run, metrics = self.evaluation_loop_iterative_vectorized(
                query_dataloader, doc_data_loader, eval_dataset.qrels)
        # else:
        # run, metrics = self.evaluation_loop_iterative(
        #     query_dataloader, doc_data_loader, eval_dataset.qrels)
        metrics = {
            f"{metric_key_prefix}_{str(m)}": metrics[m] for m in metrics}
        metrics.update(speed_metrics(metric_key_prefix,
                       start_time))
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics)
        self._memory_tracker.stop_and_update_metrics(metrics)
        return metrics

    def predict(
        self, test_dataset: Dataset, ignore_keys: Optional[List[str]] = None, metric_key_prefix: str = "test"
    ):
        """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in `evaluate()`.

        Args:
            test_dataset (`Dataset`):
                Dataset to run the predictions on. If it is an `datasets.Dataset`, columns not accepted by the
                `model.forward()` method are automatically removed. Has to implement the method `__len__`
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"test"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "test_bleu" if the prefix is "test" (default)

        <Tip>

        If your predictions or labels have different sequence length (for instance because you're doing dynamic padding
        in a token classification task) the predictions will be padded (on the right) to allow for concatenation into
        one array. The padding index is -100.

        </Tip>

        Returns: *NamedTuple* A namedtuple with the following keys:

            - predictions (`np.ndarray`): The predictions on `test_dataset`.
            - label_ids (`np.ndarray`, *optional*): The labels (if the dataset contained some).
            - metrics (`Dict[str, float]`, *optional*): The potential dictionary of metrics (if the dataset contained
              labels).
        """
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()
        if test_dataset is None:
            test_dataset = self.test_dataset
        doc_data_loader = self.get_test_dataloader(
            test_dataset.docs)
        query_dataloader = self.get_test_dataloader(
            test_dataset.queries)
        print(
            f"Prediction queries: {len(test_dataset.queries)} with {len(query_dataloader)} batches")
        print(
            f"Prediction docs: {len(test_dataset.docs)} with {len(doc_data_loader)} batches")
        start_time = time.time()
        run_path = self.args.output_dir + "/test_run.trec"
        if self.inverted_index:
            test_run, metrics = self.evaluation_loop(
                query_dataloader, doc_data_loader, test_dataset.qrels, run_path=run_path)
        else:
            test_run, metrics = self.evaluation_loop_iterative_vectorized(
                query_dataloader, doc_data_loader, test_dataset.qrels, run_path=run_path, save_representation=True)
        metrics = {
            f"{metric_key_prefix}_{str(m)}": metrics[m] for m in metrics}
        metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
            )
        )
        self.log(metrics)
        self.control = self.callback_handler.on_predict(
            self.args, self.state, self.control, metrics)
        self._memory_tracker.stop_and_update_metrics(metrics)
        result_path = self.args.output_dir + "/test_result.json"
        logger.info(f"Saving test metrics to {result_path}")
        json.dump(metrics, open(result_path, "w"))
        return PredictionOutput(predictions=test_run,  label_ids=None, metrics=metrics)
