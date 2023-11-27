import collections
import hashlib
import numpy as np
import torch
import sys
import os
import argparse

this_file_path = os.path.join(os.getcwd(), __file__)
project_path = os.path.split(os.path.split(os.path.split(this_file_path)[0])[0])[0]
print(project_path)
sys.path.insert(0, project_path)

from corpora.data import Corpus

parser = argparse.ArgumentParser()
parser.add_argument("--lang", type=str, default="es", help="corpus to base this one off of (length, vocab size etc)")
args = parser.parse_args()
print(args)
# Language to base the distribution on. Should not be that important.


lang_fn = os.path.join(project_path, "corpora", "pickled_files", f"corpus-{args.lang}.cull")
print(f"loading original lang corpus from {lang_fn}")
lang_corpus = torch.load(lang_fn)

uni_corpus = Corpus()
uni_corpus.dictionary = lang_corpus.dictionary
word_indices, freq = list(zip(*lang_corpus.dictionary.counter.items()))
freq = np.array(freq)
ps = freq / sum(freq)

uni_corpus.train = torch.LongTensor(
    np.random.choice(word_indices, len(lang_corpus.train), p=ps))
uni_corpus.valid = torch.LongTensor(
    np.random.choice(word_indices, len(lang_corpus.valid), p=ps))
uni_corpus.test = torch.LongTensor(
    np.random.choice(word_indices, len(lang_corpus.test), p=ps))

print(f"Made train/valid/test, train_length is {len(uni_corpus.train)}")

save_fn = os.path.join(project_path, "corpora", "pickled_files", f"corpus-unigram-{args.lang}.cull") 
torch.save(uni_corpus, save_fn)
