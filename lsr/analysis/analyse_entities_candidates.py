import json
import ir_datasets
import argparse
parser = argparse.ArgumentParser("Analyse doc entity")
parser.add_argument("--text", type=str,
                    default="data/robust04/queries.tsv", help="Text input")
parser.add_argument("--ent", type=str, default="/scratch-shared/tnguyen/lsr-entities/outputs/qmlp_dmlm_emlm_robust04_msmarco_pretrained_inparsv2_monot53b_distillation_l1_noent_0.0_0.001/eval/tmp_eval/docs/part_0.jsonl")
parser.add_argument("--query", action="store_true")
parser.add_argument("--ir_dataset", action="store_true")
args = parser.parse_args()
if args.query:
    qid2text = {}
    if args.ir_dataset:
        dataset = ir_datasets.load(args.text)
        for q in dataset.queries_iter():
            qid2text[q.query_id] = q.text
    else:
        with open(args.text) as f:
            for line in f:
                qid, qtext = line.strip().split("\t")
                qid2text[qid] = qtext

    def gettext(q_id):
        if not q_id in qid2text:
            return "Not text ..."
        else:
            return qid2text[q_id]
else:
    dataset = ir_datasets.load("disks45/nocr/trec-robust-2004")

    def gettext(doc_id):
        doc_text = dataset.docs_store().get(doc_id)
        doc_text = doc_text.title + " " + doc_text.body.replace("\n", " ")
        return doc_text

entity_dictionary = json.load(open("data/entity_list.json"))
with open(args.ent, "r") as f:
    for line in f:
        item = json.loads(line.strip())
        text_id = item["id"]
        text = gettext(text_id)
        print("=======================")
        print(f"ID: {text_id}\n")
        print(f"text: {text}\n\n")
        print("#######################")
        entities = [entity_dictionary[ent_id]
                    for ent_id in item["entities"][:20]]
        print(entities)
        input()
