import argparse
import collections
import hashlib
import numpy as np
import torch
import os

import sys
this_file_path = os.path.join(os.getcwd(), __file__)
project_path = os.path.split(os.path.split(os.path.split(this_file_path)[0])[0])[0]
print(project_path)
sys.path.insert(0, project_path)

import corpora.data as data

default_vocab_size = 50000
parser = argparse.ArgumentParser(description="Create a corpus from a uniform distribution")
parser.add_argument("--lang", type=str, default="ja", help="corpus to base this one off of (length, vocab size etc)")
parser.add_argument('--vocab-size', type=int, default=default_vocab_size)
args = parser.parse_args()

# Language to base the corpus length on. Should not be that important.
lang_fn = os.path.join(project_path, "corpora", "pickled_files", f"corpus-{args.lang}.cull")
lang_corpus = torch.load(lang_fn)

uni_corpus = data.Corpus()
uni_corpus.dictionary = lang_corpus.dictionary

uni_corpus.train = torch.LongTensor(
    np.random.randint(args.vocab_size, size=len(lang_corpus.train)))
uni_corpus.valid = torch.LongTensor(
    np.random.randint(args.vocab_size, size=len(lang_corpus.valid)))
uni_corpus.test = torch.LongTensor(
    np.random.randint(args.vocab_size, size=len(lang_corpus.test)))

print(f"Made train/valid/test, train_length is {len(uni_corpus.train)}")

if args.vocab_size == default_vocab_size:
    save_fn = os.path.join(project_path, "corpora", "pickled_files", f"corpus-random-{args.lang}.cull") 
else:
    save_fn = os.path.join(project_path, "corpora", "pickled_files", f"corpus-random-{args.lang}-size{args.vocab_size}.cull") 
print(f"Saving to {save_fn}")
torch.save(uni_corpus, save_fn)
