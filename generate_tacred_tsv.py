"""
Put this script under your TACRED dataset root directory. 
Run the generate_json.py script (provided by the author of TACRED dataset) first
Then, run this script to generate train/dev/test files in R-BERT input tsv format
"""

import json

splits = ["train", "dev", "test"]
cases = ["cased", "uncased"]

relation2id = {}
id2relation = []

for split in splits:
    with open(f"./data/json/{split}.json", "r") as fin:
        data = json.load(fin)
    with open(f"./data/gold/{split}.gold", "r") as fin:
        labels = [line.strip() for line in fin.readlines() if line.strip() != ""]
    assert [ele['relation'] for ele in data] == labels, f"mismatch label order in {split} split"
    
    # generate relation id only once
    if split == "train":  
        for sample in data:
            relation = sample['relation']
            if relation not in relation2id:
                id2relation.append(relation)
                relation2id[relation] = len(relation2id)

    with open(f"./data/tsv_cased/relation2id.tsv", "w") as fout:
        fout.write("\n".join(id2relation)+"\n")
    with open(f"./data/tsv_uncased/relation2id.tsv", "w") as fout:
        fout.write("\n".join(id2relation)+"\n")
    
    for case in cases:
        print(f"Process {split} splits -- {case}")
        with open(f"./data/tsv_{case}/{split}.tsv", "w") as fout:
            for idx, sample in enumerate(data):
                tokens = sample['token']
                subj_start = sample['subj_start']
                subj_end = sample['subj_end']
                subj_type = sample['subj_type']
                obj_start = sample['obj_start']
                obj_end = sample['obj_end']
                obj_type = sample['obj_type']
                relation_id = relation2id[sample['relation']]

                subject_first = (subj_start < obj_start)
                if subject_first:  # first encountered entity is subject
                    new_tokens = tokens[:subj_start] + ["[E11]"] + tokens[subj_start:subj_end+1] + ["[E12]"] + \
                        tokens[subj_end+1:obj_start] + ["[E21]"] + tokens[obj_start:obj_end+1] + ["[E22]"] + \
                        tokens[obj_end+1:]
                else:
                    new_tokens = tokens[:obj_start] + ["[E21]"] + tokens[obj_start:obj_end+1] + ["[E22]"] + \
                        tokens[obj_end+1:subj_start] + ["[E11]"] + tokens[subj_start:subj_end+1] + ["[E12]"] + \
                        tokens[subj_end+1:]

                if case == "uncased":
                    new_tokens = [t.lower() if t not in ["[E11]", "[E12]", "[E21]", "[E22]"] else t for t in new_tokens]

                token_string = " ".join(new_tokens)
                fout.write(f"{idx}\t{token_string}\t{relation_id}\t{subj_type}\t{obj_type}\n")
