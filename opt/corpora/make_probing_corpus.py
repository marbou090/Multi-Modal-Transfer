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
parser.add_argument('--batch-condition', type=str, default='batch', help='学習時のバッチサイズに合わせて<unk>埋めするか、文の最大長に合わせるか')
args = parser.parse_args()

def main(args):
    corpus = Corpus()
    corpus.train = tokenize(corpus, args.path , 'train', args.batch_condition)
    corpus.valid = tokenize(corpus, args.path, 'validation', args.batch_condition)
    corpus.test = tokenize(corpus, args.path, 'test', args.batch_condition)
    torch.save(corpus, os.path.join(project_path, "corpora", "probing_pickled_files", f"corpus-{args.name}"))
    print("Finished and saved!")

def tokenize(corpus, path, option, batch):
    if batch == 'batch':
        batch_size = 80
        return probing_tokenize_batch(corpus, path, option, batch_size)
    else:
        return probing_tokenize(corpus, path, option)

def probing_tokenize(corpus, path, option):
    """Tokenizes a text file."""
    print(path)
    assert os.path.exists(path)
    # Add words to the dictionary
    # add target
    max_line_length = 0
    lines_length = 0 #全部で何行あるのか

    mode = ''
    if option =='train':
        mode = 'tr'
    elif option == 'validation':
        mode = 'va'
    elif option == 'test':
        mode = 'te'
    
    corpus.dictionary.add_word("<unk>")

    test_sentence = ''
    #add word
    with open(path, 'r') as f:      
        tokens = 0 
        for line in f:
            sentences = line.split('\t')
            words = sentences[2].split() + ['<eos>']
            if max_line_length < len(words):
                max_line_length = len(words)
            if sentences[0] == mode:
                if len(test_sentence)<1:
                    test_sentence = sentences
                tokens += len(words)
                for word in words:
                    corpus.dictionary.add_word(word)


    # Tokenize file content
    with open(path, 'r') as f:
        ids = torch.LongTensor(tokens)
        token = 0
        for line in f:
            sentences = line.split('\t')
            words = sentences[2].split() + ['<eos>']
            if len(words)<max_line_length:
                while len(words) >= max_line_length:
                    words.append('<unk>')
            if sentences[0] == mode:
                for word in words:
                    ids[token] = corpus.dictionary.get_index(word)
                    token += 1

    
    print(f"max length : {max_line_length}")
    print('-'*89)
    print(f'{option} dataset ')
    print(f'target:{test_sentence[1]}')
    print(f'original sentence:{test_sentence[2]}')
    token_text = ''
    for word in test_sentence[2]:
        token_text = token_text + str(corpus.dictionary.get_index(word)) + ' '
    print(f'token sentence:{token_text}')
    return ids

def probing_tokenize_batch(corpus, path, option, batchsize):
    """Tokenizes a text file."""
    print(path)
    assert os.path.exists(path)
    # Add words to the dictionary
    # add target
    lines_length = 0 #全部で何行あるのか

    mode = ''
    if option =='train':
        mode = 'tr'
    elif option == 'validation':
        mode = 'va'
    elif option == 'test':
        mode = 'te'
    
    corpus.dictionary.add_word("<unk>")

    test_sentence = ''
    #add word
    with open(path, 'r') as f:      
        tokens = 0 
        for line in f:
            sentences = line.split('\t')
            words = sentences[2].split() + ['<eos>']
            if sentences[0] == mode:
                if len(test_sentence)<1:
                    test_sentence = sentences
                tokens += len(words)
                for word in words:
                    corpus.dictionary.add_word(word)


    # Tokenize file content
    with open(path, 'r') as f:
        ids = torch.LongTensor(tokens)
        token = 0
        for line in f:
            sentences = line.split('\t')
            words = sentences[2].split() + ['<eos>']
            if len(words)<batchsize:
                while len(words) >= batchsize:
                    words.append('<unk>')
            if sentences[0] == mode:
                for word in words:
                    ids[token] = corpus.dictionary.get_index(word)
                    token += 1

    
    print(f"batch size : {batchsize}")
    print('-'*89)
    print(f'{option} dataset ')
    print(f'target:{test_sentence[1]}')
    print(f'original sentence:{test_sentence[2]}')
    token_text = ''
    for word in test_sentence[2]:
        token_text = token_text + str(corpus.dictionary.get_index(word)) + ' '
    print(f'token sentence:{token_text}')
    return ids


if __name__ == "__main__":
    
    print(args)
    main(args)
