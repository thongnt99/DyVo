from tqdm import tqdm
from pathlib import Path
import ir_datasets
import ir_measures
from collections import defaultdict
import random

random.seed(42)
# loading runfiles
run_file = ir_measures.read_trec_run("data/bm25/run.robust04.trec")
q2top1k = defaultdict(list)
for pair in run_file:
    q2top1k[pair.query_id].append((pair.doc_id, pair.score))

folds = [ir_datasets.load(
    f"disks45/nocr/trec-robust-2004/fold{foldidx}") for foldidx in range(1, 6)]


def sample_triplets(fold_x, k=100):
    triplets = []
    q2pos = defaultdict(set)
    q2neg = defaultdict(set)
    no_neg = 0
    for pair in fold_x.qrels_iter():
        if pair.relevance > 0:
            q2pos[pair.query_id].add(pair.doc_id)
        # else:
            # q2neg[pair.query_id].add(pair.doc_id)
    for query_id in tqdm(q2pos):
        top10_bm25 = sorted(q2top1k[query_id],
                            key=lambda x: x[1], reverse=True)[:k]
        for doc_id, _score in top10_bm25:
            if not doc_id in q2pos[query_id]:
                q2neg[query_id].add(doc_id)
        if len(q2neg[query_id]) == 0:
            no_neg += 1
        for pos_id in q2pos[query_id]:
            for neg_id in q2neg[query_id]:
                triplets.append([query_id, pos_id, neg_id])
    random.shuffle(triplets)
    print(f"Number of queries with no negative in top {k}: {no_neg}")
    print(f"Num queries: {len(set([t[0] for t in triplets]))}")
    return triplets


print("Sampling triplets")
triplets = [sample_triplets(fold, 100) for fold in folds]

dir = Path("data/robust04/folds")
for fold_idx in tqdm(range(5)):
    ffold = dir/f"fold{fold_idx+1}.tsv"
    print(f"Writing training data to {ffold}")
    count_triplet = 0
    with open(ffold, "w") as f:
        for data in triplets[:fold_idx] + triplets[fold_idx+1:]:
            for triplet in data:
                count_triplet += 1
                f.write("\t".join(triplet)+"\n")
    print(f"{count_triplet} triplets written to {ffold}")
