# A Pytorch Implementation of R-BERT relation classification model

This is an unofficial pytorch implementation of `R-BERT` model described paper [Enriching Pre-trained Language Model with Entity Information for Relation Classification](https://arxiv.org/abs/1905.08284).


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

```
CUDA_VISIBLE_DEVICES=0 python bert.py --config config.ini
```

## Eval

### SemEval-2010

We use the official script for SemEval 2010 task-8

```
$ cd eval
$ bash test.sh
$ cat res.txt
```

## Results

### SemEval-2010

Below the Macro-F1 score

|        Model        |     Paper      |     Ours       |
| ------------------- | -------------- | -------------- |
| BERT-uncased-base   |     ----       |     88.40      |
| BERT-uncased-large  |     89.25      |     89.21      |


## Reference

1. [https://github.com/wang-h/bert-relation-classification](https://github.com/wang-h/bert-relation-classification)

2. [Enriching Pre-trained Language Model with Entity Information for Relation Classification](https://arxiv.org/abs/1905.08284).
