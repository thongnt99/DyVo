import json
inp_path = "data/entity.jsonl"
out_path = "data/entity_anserini.jsonl"
entity_list = json.load(open("data/entity_list.json", "r"))
entity2id = dict(zip(entity_list, range(len(entity_list))))
with open(inp_path, "r") as fin, open(out_path, "w") as fout:
    for line in fin:
        data = json.loads(line)
        ent_name = data["entity"]
        if ent_name in entity2id:
            ent_id = entity2id[ent_name]
            text = data["title"] + " " + data["text"]
            entity = {"id": ent_id, "contents": text}
            fout.write(json.dumps(entity)+"\n")
