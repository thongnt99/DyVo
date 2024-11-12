from pathlib import Path
from glob import glob
import json
score_dict = {}
for fn in glob("data/wapo/inparsv2/scores/*"):
    score_partx = json.load(open(fn))
    score_dict.update(score_partx)

json.dump(score_dict, open(
    "data/wapo/inparsv2/prefix_monot5_3b_logits.json", "w"))
# for qid in score_partx:
#     score_dict[qid] = score_partx[qid]
