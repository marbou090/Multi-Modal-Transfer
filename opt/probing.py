import argparse
import sys
import os
import sys
import pickle
sys.path.append("..")
print(sys.path)
project_base_path = ".."

import torch

from probing_method.conllu_utils import load, tokenize, data_loader

TRAIN_FILE = "corpora/UD_English-EWT/en_ewt-ud-train.conllu"
EVAL_FILE = "corpora/UD_English-EWT/en_ewt-ud-dev.conllu"
TEST_FILE = "corpora/UD_English-EWT/en_ewt-ud-test.conllu"

parser = argparse.ArgumentParser()
parser.add_argument('--finetuned-model', help='the pretrained model to probe')
parser.add_argument('--finetuned-data', help='')
parser.add_argument('--run-name', type=str, default="last_run", help="How to call the save file for this run, within the results dir of this pretrain type")
parser.add_argument('--task', default='Case', help='The probing task to execute.')
parser.add_argument('--seed', default=400, help='The seed used for the Numpy RNG.')
parser.add_argument('--probe', help="Which probing model to use.", default="linear", choices=['mlp', 'linear'])
parser.add_argument('--num-repetitions', default=5)
args = parser.parse_args()
args.cuda = False
print(args)

possible_probetask = \
    ['Case','Definite','Degree','ExtPos','Gender','Mood','Number','NumForm','NumType','Person','PronType','Tense','VerbForm']
assert args.task in possible_probetask

batch_size = 80
test_batch_size = 1

def run():
    train_sentences, train_labels = load(TRAIN_FILE)
    eval_sentences, eval_labels = load(EVAL_FILE)
    test_sentences, test_labels = load(TEST_FILE)

    dict_label={}
    dict_word={}
    train_sentences_ids, train_tagging_ids, dict_word,dict_label = tokenize(train_sentences, train_labels, dict_word,dict_label)
    eval_sentences_ids, eval_tagging_ids, dict_word,dict_label = tokenize(eval_sentences, eval_labels, dict_word,dict_label)
    test_sentences_ids, test_tagging_ids,dict_word,dict_label = tokenize(test_sentences, test_labels, dict_word,dict_label)

    train_loader = data_loader(train_sentences_ids, train_tagging_ids, batch_size)
    eval_loader = data_loader(eval_sentences_ids, eval_tagging_ids, batch_size)
    test_loader = data_loader(test_sentences_ids, test_tagging_ids, batch_size)

    if args.model == 'linear':
        probemodel = LinearModel()
    
    embeddings = compute_embeddings(embedding_model)
    


if __name__ == "__main__":
    run()
