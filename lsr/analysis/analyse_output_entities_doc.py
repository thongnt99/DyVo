import json
import ir_datasets
import argparse
parser = argparse.ArgumentParser("Parsing arguments")
parser.add_argument("--d", type=str, default="disks45/nocr/trec-robust-2004")
parser.add_argument("--d_ent", type=str, default="/scratch-shared/tnguyen/lsr-entities/outputs/qmlp_dmlm_emlm_robust04_msmarco_pretrained_inparsv2_monot53b_distillation_l1_noent_0.0_0.001/eval/tmp_eval/docs/part_0.jsonl")
args = parser.parse_args()
dataset = ir_datasets.load(args.d)
entity_dictionary = json.load(open("data/entity_list.json"))
inp_path = args.d_ent
with open("msmarco_entities_analysis.txt", "w") as fout:
    with open(inp_path, "r") as f:
        for line in f:
            doc = json.loads(line.strip())
            doc_id = doc["id"]
            token2weights = doc["vector"]
            entities = []
            for token in token2weights:
                int_token = int(token)
                if int_token >= 30522:
                    entity_id = int_token - 30522
                    entities.append(
                        (entity_id, entity_dictionary[entity_id], token2weights[token]))
            entities = sorted(entities, key=lambda x: x[2], reverse=True)
            doc = dataset.docs_store().get(doc_id)
            doc_text = str(doc.title) + doc.body.replace("\n", " ")
            fout.write("=======================")
            fout.write(f"doc_id: {doc_id}\n")
            fout.write(f"text: {doc_text}\n\n")
            for ent in entities[-10:]:
                fout.write(f"{ent[0]}\t{ent[1]}\t{ent[2]}\n")
