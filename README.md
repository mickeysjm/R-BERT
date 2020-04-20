# A Pytorch Implementation of R-BERT relation classification model

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/enriching-pre-trained-language-model-with/relation-extraction-on-semeval-2010-task-8)](https://paperswithcode.com/sota/relation-extraction-on-semeval-2010-task-8?p=enriching-pre-trained-language-model-with)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/enriching-pre-trained-language-model-with/relation-extraction-on-tacred)](https://paperswithcode.com/sota/relation-extraction-on-tacred?p=enriching-pre-trained-language-model-with)

This is an unofficial pytorch implementation of `R-BERT` model described paper [Enriching Pre-trained Language Model with Entity Information for Relation Classification](https://arxiv.org/abs/1905.08284).

In addition to the SemEval 2010 dataset tested in the original paper, I aslo test implementation on the more recent [TACRED](https://nlp.stanford.edu/projects/tacred/) dataset 

## Requirements:
 
- Python version >= 3.6
- Pytorch version >= 1.1
- [Transformer library](https://github.com/huggingface/transformers) version >= 2.5.1

## Install

```
$ https://github.com/mickeystroller/R-BERT
$ cd R-BERT
```

## Train

### SemEval-2010 

The SemEval-2010 dataset is already included in this repo and you can directly run:

```
CUDA_VISIBLE_DEVICES=0 python r_bert.py --config config.ini
```

### TACRED

You need to first download TACRED dataset from [LDC](https://catalog.ldc.upenn.edu/LDC2018T24), which due to the license issue I cannot put in this repo. Then, you can directly run:

```
CUDA_VISIBLE_DEVICES=0 python r_bert.py --config config_tacred.ini
```

## Eval

### SemEval-2010

We use the official script for SemEval 2010 task-8

```
$ cd eval
$ bash test.sh
$ cat res.txt
```

### TACRED

First, we generate prediction file `tac_res.txt` 

```
$ python eval_tacred.py
```

You may change test file/model path in the `eval_tacred.py` file

Then, we use the official scoring script for TACRED dataset

```
$ python ./eval/score.py -gold_file <TACRED_DIR/data/gold/test.gold> -pred_file ./eval/tac_res.txt
```


## Results

### SemEval-2010

Below is the Macro-F1 score

|        Model        | Original Paper |     Ours       |
| ------------------- | -------------- | -------------- |
| BERT-uncased-base   |     ----       |     88.40      |
| BERT-uncased-large  |     89.25      |    **90.16**   |


### TACRED

Below is the evaluation result

|        Model        |  Precision (Micro) | Recall (Micro) | F1 (Micro) |   
| ------------------- | ------------------ | -------------- | ---------- |
| BERT-uncased-base   |    **72.99**       |   62.50        |    67.34   |
| BERT-cased-base     |    71.27           |   64.84        |    67.91   |
| BERT-uncased-large  |    72.91           |   **66.20**    |  **69.39** |
| BERT-cased-large    |    70.86           |   65.96        |    68.32   |



## Reference

1. [https://github.com/wang-h/bert-relation-classification](https://github.com/wang-h/bert-relation-classification)

2. [Enriching Pre-trained Language Model with Entity Information for Relation Classification](https://arxiv.org/abs/1905.08284).
