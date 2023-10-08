# Make a Corpus object from a clean, tokenized corpus file. If you have some 
# downloaded dump, first run a script from raw_to_tokens and then come here.
import argparse
import re
import os
import pickle
import torch
import numpy as np

import sys
this_file_path = os.path.join(os.getcwd(), __file__)
project_path = os.path.split(os.path.split(this_file_path)[0])[0]
print(project_path)
sys.path.insert(0, project_path)

from corpora.data import Corpus

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help="Path of where the data is, minus the train/val/test part of the filename")
parser.add_argument('--name', type=str, help="The shorthand name for this corpus, such as 'unigram' or 'pt'")

def main(args):
    corpus = Corpus()
    corpus.train = tokenize(corpus, args.path + 'train')
    corpus.valid = tokenize(corpus, args.path + 'validation')
    corpus.test = tokenize(corpus, args.path + 'test')
    torch.save(corpus, os.path.join(project_path, "corpora", "pickled_files", f"corpus-{args.name}"))
    print("Finished and saved!")

def tokenize(corpus, path):
    return probing_tokenize(corpus, path)

def probing_tokenize(corpus, path):
    """Tokenizes a text file."""
    print(path)
    assert os.path.exists(path)
    # Add words to the dictionary
    with open(path, 'r') as f:
        tokens = 0
        for line in f:
            line = re.sub('\A[A-Z][a-z][a-z]','',line)
            words = line.split() + ['<eos>']
            tokens += len(words)
            for word in words:
                corpus.dictionary.add_word(word)

    # Tokenize file content
    with open(path, 'r') as f:
        ids = torch.LongTensor(tokens)
        token = 0
        for line in f:
            line = re.sub('\A[A-Z][a-z][a-z]','',line)
            words = line.split() + ['<eos>']
            for word in words:
                ids[token] = corpus.dictionary.get_index(word)
                token += 1
    return ids

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    main(args)
