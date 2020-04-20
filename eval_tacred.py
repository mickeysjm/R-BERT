import sys
import numpy as np
import torch
from torch.utils.data import (DataLoader, SequentialSampler, TensorDataset)
from tqdm import tqdm
from transformers import BertTokenizer
from utils import (TACRED_RELATION_LABELS, compute_metrics, 
    convert_examples_to_features, InputExample)
from model import BertForSequenceClassification
import csv


def _read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            if sys.version_info[0] == 2:
                line = list(cell for cell in line)
            lines.append(line)
        return lines


def _create_examples(lines, set_type):
    """Creates examples for the training and dev sets.
    e.g.,: 
    2    the [E11] author [E12] of a keygen uses a [E21] disassembler [E22] to look at the raw assembly code .    6
    """
    examples = []
    for (i, line) in enumerate(lines):
        guid = "%s-%s" % (set_type, i)
        text_a = line[1]
        text_b = None
        label = line[2]
        examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


def load_examples(input_file, tokenizer, max_seq_len=192, n_labels=42):
    """ A simplified version of data loading 
    
    input_file: str, a tsv file for to be predicted data
    tokenizer: tokenizer
    max_seq_len: int, max sequence length
    n_labels: int, number of labels
    """
    label_list = [str(i) for i in range(n_labels)]
    
    # Load data features from dataset file
    lines = _read_tsv(input_file)
    examples = _create_examples(lines, "eval")
    print(len(examples))
    features = convert_examples_to_features(
        examples, label_list, max_seq_len, tokenizer, "classification", use_entity_indicator=True)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in features], dtype=torch.long)
    all_e1_mask = torch.tensor(
        [f.e1_mask for f in features], dtype=torch.long)  # add e1 mask
    all_e2_mask = torch.tensor(
        [f.e2_mask for f in features], dtype=torch.long)  # add e2 mask
    all_label_ids = torch.tensor(
        [f.label_id for f in features], dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask,
                            all_segment_ids, all_label_ids, all_e1_mask, all_e2_mask)
    return dataset


"""
Setup your parameters 
"""
n_gpu = 1
device = torch.device("cuda:5")
# checkpoint = "./output/tacred/checkpoint-12000/"
# pretrained_model_name = "bert-base-cased"
checkpoint = "./output/tacred-large/checkpoint-12000/"
pretrained_model_name = "bert-large-cased"
do_lower = ("-uncased" in pretrained_model_name)
input_file = "/home/jiaming/datasets/TACRED/data/tsv_cased/test.tsv"
# output_eval_file = "./eval/tac_res.txt"
output_eval_file = "./eval/tac_res_large.txt"
batch_size=16

"""
Start eval
"""
additional_special_tokens = ["[E11]", "[E12]", "[E21]", "[E22]"]
tokenizer = BertTokenizer.from_pretrained(pretrained_model_name, 
    do_lower_case=do_lower, additional_special_tokens=additional_special_tokens)
model = BertForSequenceClassification.from_pretrained(checkpoint)
model.to(device)

eval_dataset = load_examples(input_file, tokenizer)
eval_sampler = SequentialSampler(eval_dataset)
eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=batch_size, shuffle=False)

eval_loss = 0.0
nb_eval_steps = 0
pred_logits = None
out_label_ids = None
input_ids = None

for batch in tqdm(eval_dataloader, desc="Evaluating"):
    model.eval()
    batch = tuple(t.to(device) for t in batch)

    with torch.no_grad():
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                  # XLM and RoBERTa don't use segment_ids
                  'token_type_ids': batch[2],
                  'labels':      batch[3],
                  'e1_mask': batch[4],
                  'e2_mask': batch[5],
                  }
        outputs = model(**inputs)
        tmp_eval_loss, logits = outputs[:2]

        eval_loss += tmp_eval_loss.mean().item()
    nb_eval_steps += 1
    if pred_logits is None:
        pred_logits = logits.detach().cpu().numpy()
        out_label_ids = inputs['labels'].detach().cpu().numpy()
    else:
        pred_logits = np.append(pred_logits, logits.detach().cpu().numpy(), axis=0)
        out_label_ids = np.append(
            out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

preds = np.argmax(pred_logits, axis=1)
result = compute_metrics("classification", preds, out_label_ids)
for key in sorted(result.keys()):
    print(f"{key} = {result[key]}")

with open(output_eval_file, "w") as writer:
    for pred in preds:
        writer.write(TACRED_RELATION_LABELS[pred])
        writer.write("\n")
