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

from corpora.probing_data import Corpus

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help="Path of where the data is, minus the train/val/test part of the filename")
parser.add_argument('--name', type=str, help="The shorthand name for this corpus, such as 'unigram' or 'pt'")

def main(args):
    corpus = Corpus()
    corpus.train = tokenize(corpus, args.path + 'train')
    corpus.valid = tokenize(corpus, args.path + 'validation')
    corpus.test = tokenize(corpus, args.path + 'test')
    torch.save(corpus, os.path.join(project_path, "corpora", "probing_pickled_files", f"corpus-{args.name}"))
    print("Finished and saved!")

def tokenize(corpus, path):
    return probing_tokenize(corpus, path)

def probing_tokenize(corpus, path):
    """Tokenizes a text file."""
    print(path)
    assert os.path.exists(path)
    # Add words to the dictionary
    # add target
    max_line_length = 0
    lines_length = 0 #全部で何行あるのか
    with open(path, 'r') as f:
        tokens = 0
        for line in f:
            lines_length +=1
            word = re.search(r'[A-Z][a-z]+',line)
            corpus.dictionary.add_word(word.group())
    

    # add special token
    corpus.dictionary.add_word("<unk>")
    tokens=4

    #add word
    with open(path, 'r') as f:       
        for line in f:
            line = re.sub('\A[A-Z][a-z]+','',line)
            words = line.split() + ['<eos>']
            if max_line_length < len(words):
                max_line_length = len(words)
            tokens += len(words)
            for word in words:
                corpus.dictionary.add_word(word)

   
    # Tokenize file content
    with open(path, 'r') as f:
        ids = torch.LongTensor(lines_length, max_line_length+1)
        print(ids.size())
        number_line=0 #いま何行目か

        for line in f:
            words = line.split() + ['<eos>']
            if len(words)<max_line_length:
                while len(words) >= max_line_length:
                    words.append('<unk>')
            
            for i, word in enumerate(words):
                ids[number_line][i] = corpus.dictionary.get_index(word)
            number_line+=1
    
    for target in ["Acc", "Gen","Nom"]:
            print(f'target:{target}')
            print(f'token:{corpus.dictionary.get_index(target)}')
            print('-'*89)
    return ids

if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    main(args)
