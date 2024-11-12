import os
import re
import csv
import json
import random
import argparse
import subprocess
from tqdm import tqdm
from pathlib import Path
from transformers import set_seed
from pyserini.search.lucene import LuceneSearcher

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--qrel', type=str)
    parser.add_argument('--index', type=str, default='msmarco-passage')
    parser.add_argument('--max_hits', type=int, default=1000)
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument("--batch_size", default=32, type=int,
                        help="Batch size retrieval.")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    index = args.index

    if os.path.isdir(index):
        searcher = LuceneSearcher(index)
    else:
        searcher = LuceneSearcher.from_prebuilt_index(index)

    tmp_run = f'{Path(args.output).parent}/tmp-run-{args.dataset.replace("/","_")}.txt'
    if not os.path.exists(tmp_run):
        subprocess.run([
            'python3', '-m', 'pyserini.search.lucene',
            '--threads', '190',
            '--batch-size', str(args.batch_size),
            '--index', index,
            '--topics', f'{args.input}',
            '--output', tmp_run,
            '--bm25',
        ])

    results = {}
    with open(tmp_run) as f:
        for line in f:
            qid, _, docid, rank, score, ranker = re.split(r"\s+", line.strip())
            if qid not in results:
                results[qid] = []
            results[qid].append(docid)

    q2pos = json.load(open(args.qrel))
    with open(args.output, 'w') as fout:
        writer = csv.writer(fout, delimiter='\t',
                            lineterminator='\n', quoting=csv.QUOTE_MINIMAL)
        for qid in tqdm(results, desc='Sampling'):
            hits = results[qid]
            pos_doc_id = list(q2pos[qid].keys())[0]
            sampled_ranks = random.sample(
                range(len(hits)), min(len(hits), args.n_samples + 1))
            n_samples_so_far = 0
            for (rank, neg_doc_id) in enumerate(hits):
                if rank not in sampled_ranks:
                    continue
                if pos_doc_id == neg_doc_id:
                    continue
                writer.writerow([qid, pos_doc_id, neg_doc_id])
                n_samples_so_far += 1
                if n_samples_so_far >= args.n_samples:
                    break
    print("Done!")
