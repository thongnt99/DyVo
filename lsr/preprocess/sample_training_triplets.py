import json
from collections import defaultdict
import argparse
import ir_measures
import random
parser = argparse.ArgumentParser("Parsing arguments")
parser.add_argument("--run", type=str, help="Path to run file")
parser.add_argument("--qrel", type=str, help="Path to qrel file")
parser.add_argument("--output", type=str,
                    help="Path to store resulting triplets")
args = parser.parse_args()
q2hits = defaultdict(list)

for row in ir_measures.read_trec_run(args.run):
    q2hits[row.query_id].append(row.doc_id)

qrels = json.load(open(args.qrel))
with open(args.output, "w") as fout:
    for qid in qrels:
        pos_id = qrels[qid]
        hits = q2hits[qid]
        try:
            neg_samples = random.sample(hits, k=100)
            for neg_id in neg_samples:
                if neg_id != pos_id:
                    fout.write(f"{qid}\t{pos_id}\t{neg_id}\n")
        except:
            print(len(hits))
