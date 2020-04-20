# coding=utf-8
# Copyright 2020 Jiaming Shen, University of Illinois at Urbana-Champaign, Data Mining Group.
# Copyright 2019 Hao WANG, Shanghai University, KB-NLP team.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import absolute_import, division, print_function

import csv
import re
import logging
import os
import sys
from io import open
import math

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

logger = logging.getLogger(__name__)

# Used for SemEval dataset
SEMEVAL_RELATION_LABELS = ['Other', 'Message-Topic(e1,e2)', 'Message-Topic(e2,e1)',
                   'Product-Producer(e1,e2)', 'Product-Producer(e2,e1)',
                   'Instrument-Agency(e1,e2)', 'Instrument-Agency(e2,e1)',
                   'Entity-Destination(e1,e2)', 'Entity-Destination(e2,e1)',
                   'Cause-Effect(e1,e2)', 'Cause-Effect(e2,e1)',
                   'Component-Whole(e1,e2)', 'Component-Whole(e2,e1)',
                   'Entity-Origin(e1,e2)', 'Entity-Origin(e2,e1)',
                   'Member-Collection(e1,e2)', 'Member-Collection(e2,e1)',
                   'Content-Container(e1,e2)', 'Content-Container(e2,e1)']

# Used for TACRED dataset
TACRED_RELATION_LABELS = ['org:founded_by', 'no_relation', 'per:employee_of', 'org:alternate_names', 
    'per:cities_of_residence', 'per:children', 'per:title', 'per:siblings', 'per:religion', 
    'per:age', 'org:website', 'per:stateorprovinces_of_residence', 'org:member_of', 
    'org:top_members/employees', 'per:countries_of_residence', 'org:city_of_headquarters', 'org:members', 
    'org:country_of_headquarters', 'per:spouse', 'org:stateorprovince_of_headquarters', 
    'org:number_of_employees/members', 'org:parents', 'org:subsidiaries', 'per:origin', 
    'org:political/religious_affiliation', 'per:other_family', 'per:stateorprovince_of_birth', 
    'org:dissolved', 'per:date_of_death', 'org:shareholders', 'per:alternate_names', 'per:parents', 
    'per:schools_attended', 'per:cause_of_death', 'per:city_of_death', 'per:stateorprovince_of_death', 
    'org:founded', 'per:country_of_birth', 'per:date_of_birth', 'per:city_of_birth', 'per:charges', 
    'per:country_of_death']


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 e11_p, e12_p, e21_p, e22_p,
                 e1_mask, e2_mask,
                 segment_ids,
                 label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.e11_p = e11_p
        self.e12_p = e12_p
        self.e21_p = e21_p
        self.e22_p = e22_p
        self.e1_mask = e1_mask
        self.e2_mask = e2_mask


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(cell for cell in line)
                lines.append(line)
            return lines


class SemEvalProcessor(DataProcessor):
    """Processor for the SemEval-2010 data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(
            os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return [str(i) for i in range(19)]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets.
        e.g.,: 
        2	the [E11] author [E12] of a keygen uses a [E21] disassembler [E22] to look at the raw assembly code .	6
        """
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            text_b = None
            label = line[2]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class TacredProcessor(DataProcessor):
    """Processor for the TACRED data set. """

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(
            os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return [str(i) for i in range(42)]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets.
        e.g.,: 
        2   the [E11] author [E12] of a keygen uses a [E21] disassembler [E22] to look at the raw assembly code .   6
        """
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            text_b = None
            label = line[2]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def convert_examples_to_features(examples, label_list, max_seq_len,
                                 tokenizer, output_mode,
                                 cls_token='[CLS]',
                                 cls_token_segment_id=1,
                                 sep_token='[SEP]',
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 sequence_b_segment_id=1,
                                 mask_padding_with_zero=True,
                                 use_entity_indicator=True):
    """ Loads a data file into a list of `InputBatch`s
        Default, BERT/XLM pattern: [CLS] + A + [SEP] + B + [SEP]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens_a = tokenizer.tokenize(example.text_a)
        tokens_b = None

        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3". " -4" for RoBERTa.
            special_tokens_count = 3
            _truncate_seq_pair(tokens_a, tokens_b,
                               max_seq_len - special_tokens_count)
        else:
            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = 2
            if len(tokens_a) > max_seq_len - special_tokens_count:
                tokens_a = _truncate_seq(tokens_a, max_seq_len - special_tokens_count)
                
        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if tokens_b:
            tokens += tokens_b + [sep_token]
            segment_ids += [sequence_b_segment_id] * (len(tokens_b) + 1)

        tokens = [cls_token] + tokens
        segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # entity mask
        if use_entity_indicator:
            if "[E22]" not in tokens or "[E12]" not in tokens:  # remove this sentence because after max length truncation, the one entity boundary is broken
                logger.warning(f"*** Example-{ex_index} is skipped ***")
                continue 
            else:
                e11_p = tokens.index("[E11]")+1
                e12_p = tokens.index("[E12]")
                e21_p = tokens.index("[E21]")+1
                e22_p = tokens.index("[E22]")

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token] * padding_length)
        input_mask = input_mask + \
            ([0 if mask_padding_with_zero else 1] * padding_length)
        segment_ids = segment_ids + \
            ([pad_token_segment_id] * padding_length)
        if use_entity_indicator:
            e1_mask = [0 for i in range(len(input_mask))]
            e2_mask = [0 for i in range(len(input_mask))]
            for i in range(e11_p, e12_p):
                e1_mask[i] = 1
            for i in range(e21_p, e22_p):
                e2_mask[i] = 1

        assert len(input_ids) == max_seq_len, f"Error in sample: {ex_index}, len(input_ids)={len(input_ids)}"
        assert len(input_mask) == max_seq_len
        assert len(segment_ids) == max_seq_len

        if output_mode == "classification":
            # label_id = label_map[example.label]
            label_id = int(example.label)
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" %
                        " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" %
                        " ".join([str(x) for x in input_mask]))
            if use_entity_indicator:
                logger.info("e11_p: %s" % e11_p)
                logger.info("e12_p: %s" % e12_p)
                logger.info("e21_p: %s" % e21_p)
                logger.info("e22_p: %s" % e22_p)
                logger.info("e1_mask: %s" %
                            " ".join([str(x) for x in e1_mask]))
                logger.info("e2_mask: %s" %
                            " ".join([str(x) for x in e2_mask]))
            logger.info("segment_ids: %s" %
                        " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          e11_p=e11_p,
                          e12_p=e12_p,
                          e21_p=e21_p,
                          e22_p=e22_p,
                          e1_mask=e1_mask,
                          e2_mask=e2_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))

    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def _truncate_seq(tokens_a, max_length):
    """Truncates a sequence """
    tmp = tokens_a[:max_length]
    if ("[E12]" in tmp) and ("[E22]" in tmp):
        return tmp
    else:
        e11_p = tokens_a.index("[E11]")
        e12_p = tokens_a.index("[E12]")
        e21_p = tokens_a.index("[E21]")
        e22_p = tokens_a.index("[E22]")
        start = min(e11_p, e12_p, e21_p, e22_p)
        end = max(e11_p, e12_p, e21_p, e22_p)
        if end-start > max_length:
            remaining_length = max_length - (e12_p-e11_p+1) - (e22_p-e21_p+1)  
            first_addback = math.floor(remaining_length/2)
            second_addback = remaining_length - first_addback
            if start == e11_p:
                new_tokens = tokens_a[e11_p: e12_p+1+first_addback] + tokens_a[e21_p-second_addback:e22_p+1]
            else:
                new_tokens = tokens_a[e21_p: e22_p+1+first_addback] + tokens_a[e11_p-second_addback:e12_p+1]
            return new_tokens
        else:
            new_tokens = tokens_a[start:end+1]
            remaining_length = max_length - len(new_tokens)
            if start < remaining_length:  # add sentence beginning back
                new_tokens = tokens_a[:start] + new_tokens 
                remaining_length -= start
            else:
                new_tokens = tokens_a[start-remaining_length:start] + new_tokens
                return new_tokens

            # still some room left, add sentence end back
            new_tokens = new_tokens + tokens_a[end+1:end+1+remaining_length]
            return new_tokens


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels, average='micro'):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average='micro')
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
    }


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    return acc_and_f1(preds, labels)


data_processors = {
    "semeval": SemEvalProcessor,
    "tacred": TacredProcessor,
}

output_modes = {
    "semeval": "classification",
    "tacred": "classification",
}

GLUE_TASKS_NUM_LABELS = {
    "semeval": 19,
    "tacred": 42,
}
