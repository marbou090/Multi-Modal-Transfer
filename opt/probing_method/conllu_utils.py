#https://github.com/kulgg/ProbingPretrainedLM/blob/main/src/train_pos.py
import collections
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.utils.data as data_utils
import sys

import conllu

# notice that all beginning entities have an odd index
ner_labels = { 9:'<pad>',  0:'O',  1:'B-PER',  2:'I-PER',  3:'B-ORG',  4:'I-ORG',  5:'B-LOC',  6:'I-LOC',  7:'B-MISC',  8:'I-MISC'}
ner_label_length = 10

DBG_PRINT = False

def load_conllu(filename):
  with open(filename, encoding="utf-8") as fp:
    data = conllu.parse(fp.read())
  sentences = [[token['form'] for token in sentence] for sentence in data]
  taggings = [[token['xpos'] for token in sentence] for sentence in data]
  return sentences, taggings


def load(filename):
    filePath = filename
    sentences, labels = load_conllu(filePath)
    #debug_print(list(zip(sentences[0], labels[0])))
    return sentences, labels


def collate_fn(items):
  # items = [(tensor([ 101, 7384, ...19,  100]), tensor([ 0,  1,  6, ..., 11,  0])), ...]
  # max word length of sentences
  max_len = max(len(item[0]) for item in items)

  # sentences = tensor([[0, 0, 0,  ..., 0, 0, 0],[...]...]
  sentences = torch.zeros((len(items), max_len), device=items[0][0].device).long().to(device)
  # taggings = tensor([[0, 0, 0,  ..., 0, 0, 0]
  taggings = torch.zeros((len(items), max_len)).long().to(device)

  for i, (sentence, tagging) in enumerate(items):
    # end of sentences contains tensor contains zeros if len < max_len
    sentences[i][0:len(sentence)] = sentence
    taggings[i][0:len(tagging)] = tagging

  return sentences, taggings

def data_loader(sentences_ids, taggings_ids, batch_size):
    ds = TaggingDataset(sentences_ids, taggings_ids)
    return DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

class TaggingDataset(Dataset):
  def __init__(self, sentences, taggings):
    assert len(sentences) == len(taggings)
    self.sentences = sentences
    self.taggings = taggings

  def __getitem__(self, i):
    return self.sentences[i], self.taggings[i]

  def __len__(self):
    return len(self.sentences)
  

def tokenize(sentences, labels,dict_word,dict_label):
  """
  sentences, labels: size=12544, type=list
  sentences[0]:['Al', '-', 'Zaman', ':', 'American', 'forces', 'killed', 'Shaikh', 'Abdullah', 'al', '-', 'Ani', ',', 'the', 'preacher', 'at', 'the', 'mosque', 'in', 'the', 'town', 'of', 'Qaim', ',', 'near', 'the', 'Syrian', 'border', '.']
  labels[o]:['NNP', 'HYPH', 'NNP', ':', 'JJ', 'NNS', 'VBD', 'NNP', 'NNP', 'NNP', 'HYPH', 'NNP', ',', 'DT', 'NN', 'IN', 'DT', 'NN', 'IN', 'DT', 'NN', 'IN', 'NNP', ',', 'IN', 'DT', 'JJ', 'NN', '.']
  """
  length = len(sentences)

  token_sentences = []
  token_label = []
  word_count = 0
  label_count = 0
  for k in range(length):
    token_w = []
    token_l = []
    for word in sentences[k]:
      if not word in dict_word:
        dict_word[word] = word_count
        word_count +=1
      token_w.append(dict_word[word])
    for label in labels[k]:
      if not label in dict_label:
        dict_label[label] = label_count
        label_count+=1
      token_l.append(dict_label[label])
    token_sentences.append(token_w)
    token_label.append(token_l)
    
  print(f'original text:{sentences[0]}')
  print(f'token text:{token_sentences[0]}')
  print(f'original label:{labels[0]}')
  print(f'token label:{token_label[0]}')
  print('-'*60)

  return token_sentences, token_label, dict_word, dict_label




