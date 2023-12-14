# This is the pretrain.py from the tilt-transfer repository
import argparse
import numpy as np
import os
import csv
import json
import collections
import time
from tqdm import tqdm

import torch
from torch import nn

from paths import project_base_path
from training.utils import repackage_hidden
from probing_method.probing_model import MultiLayerProbingModel,LinearProbingModel


parser = argparse.ArgumentParser()

parser.add_argument('--probe-task', type=str, default='bigram',
                    help='Short name of the probe task')
parser.add_argument('--probe-model', type=str, default='MLP',
                    help='type of probe model (MLP, Linear)')
parser.add_argument('--pretrain-data', type=str, default='es',
                    help='short name of the corpus')
parser.add_argument('--pretrain-model', type=str, default='last_run',
                    help='what name is the save file for using run')
parser.add_argument('--seed', type=int, default=1111)

parser.add_argument('--lr', type=float, default=40,
                    help='initial learning rate')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--lr-patience', type=float, default=5,
                    help='How many times to wait in same learning rate for no improvement')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--valid-interval', type=int, default=1000,
                    help='At how many batches to check validation error and save/etc')
parser.add_argument('--activation', type=str, default='None')
args = parser.parse_args()

np.random.seed(args.seed)
print(f"Set the seed to {args.seed}")

###############################################################################
# Function
###############################################################################

#####Data
def model_load(fn):
    with open(fn, 'rb') as f:
        model, criterion, optimizer, scheduler, run_data = torch.load(f)
    return model, criterion, optimizer, scheduler, run_data

def model_save(fn, model):
    with open(fn, 'wb') as f:
        torch.save([model], f)

def probe_data_load(probe_data, max_length):
    nbatch = probe_data.size(0) // max_length
    probe_data = probe_data.narrow(0, 0, nbatch * max_length)
    probe_data = probe_data.view(max_length, -1).t().contiguous()
    return probe_data

def cul_sentence_max_length(fn):
    max_length = 0
    with open(fn, 'r') as f:
        for line in f:
            sentences = line.split('\t')
            words = sentences[2].split() + ['<eos>']
            if max_length < len(words):
                max_length = len(words)
    return max_length

def create_probe_data(fn):    
    return train_X, test_X, val_X

def probe_label_load(fn, clip_size):    
    train_data = []
    val_data = []
    test_data = []
    classes = {}
    num_classes = 0
    count = 0
    with open(fn, 'r') as f:
        for line in f:
            sentences = line.split('\t')
            if not sentences[1] in classes:
                classes[sentences[1]] = num_classes
                num_classes = num_classes + 1
    with open(fn, 'r') as f:
        index = 0
        for line in f:
            sentences = line.split('\t')
            if sentences[0] == ("tr") and len(train_data) < clip_size:
                count += 1
                train_data.append(classes[sentences[1]])
            elif sentences[0] == ("va") and len(val_data) < clip_size:
                val_data.append(classes[sentences[1]])
            elif sentences[0] == ("te") and len(test_data) < clip_size:
                test_data.append(classes[sentences[1]])
            index += 1
    return torch.tensor(train_data), torch.tensor(val_data), torch.tensor(test_data), num_classes

def model_representation(model, train_data, clip_size, batch_size=80):
    hidden = model.init_hidden(batch_size)
    index = 0
    bptt = batch_size

    for index in tqdm(range(0, train_data.size(0), bptt)):
        if batch_size > train_data.size(0)-index:
            break
        if index >= clip_size:
            break        
        data = train_data[index:index+bptt]
        hidden = repackage_hidden(hidden)
        if index == 0:
            representations, _, _, _ = model(data, hidden, return_h=True)#.view(bptt, -1)
            representations = representations.view(bptt, -1)
        else:
            representation, _, _, _ = model(data, hidden, return_h=True)
            representations = torch.cat((representations,representation.view(bptt, -1)), dim = 0)
    return representations

#####Model
def load_pret_model(fn):
    with open(fn, 'rb') as f:
        model, _, _, _, _ = torch.load(f)
    return model


#####Train
def train(model, criterion, data_X, data_Y):
    model.train()
    optimizer.zero_grad()
    predictions = model(data_X)
    print(predictions[:10])
    print(predictions.size())
    loss = criterion(predictions, data_Y)
    loss.backward()
    optimizer.step()

    """
        if epoch_batch % args.log_interval == 0 and epoch_batch > 0:
            cur_loss = total_loss.item() / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                    'tr loss {:5.2f}'.format(
                epoch, epoch_batch,  num_data // len(data_X), optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss))
            total_loss = 0
            start_titorch.load(fn)
            me = time.time()
        
        if overall_batch % args.valid_interval == 0 and overall_batch > 0:
            val_loss = evaluate(val_data, eval_batch_size)
            val_loss_list.append(val_loss)
            scheduler.step(val_loss)
            print('-'*89)
            print('| validating at batch {:3d} | time: {:5.2f}m | valid loss {:5.2f}'.format(
              overall_batch, elapsed / 60, val_loss))
            print('-'*89)
            valid_time = time.time()
        
        epoch_batch += 1
        overall_batch += 1
        epoch_data_index += seq_len
    epoch_batch = 0
    epoch_data_index = 0
    """
    return loss

###############################################################################
clip_size = 2000

# Load dataprobe_task
probe_path = os.path.join(project_base_path, 'corpora', 'SentEval','bigram_shift.txt')
train_Y, val_Y, test_Y, num_classe = probe_label_load(probe_path, clip_size)
print(f'num class : {num_classe}')
print(f"train_Y size : {len(train_Y)} | val_Y size : {len(val_Y)} | test_Y size : {len(test_Y)}")

batch_size = cul_sentence_max_length(probe_path)
batch_size = 80
corpus_path = os.path.join(project_base_path, "corpora","probing_pickled_files",'corpus-'+args.probe_task)
corpus = torch.load(corpus_path)
train_corpus = probe_data_load(corpus.train, batch_size)
val_corpus = probe_data_load(corpus.valid, batch_size)
test_corpus = probe_data_load(corpus.test, batch_size)
print(f'data length : {len(corpus.train + len(corpus.valid)+ len(corpus.test))}')
print(f'train size : {train_corpus.size()}\n validation size : {val_corpus.size()}\n test size : {test_corpus.size()}')
print(f'train corpus size : {len(corpus.train)}\n validation corpus size : {len(corpus.valid)}\n test corpus size : {len(corpus.test)}')


print("Generating Probe Task Data Set")
#学習済みに文章いれて内部表現を保存していく
print(train_corpus[0])
pret_model = load_pret_model(os.path.join(project_base_path, "models", "l2_results", "RNN", args.pretrain_data, args.pretrain_model+'-finetune.pickle'))
pret_model = pret_model.to('cpu')


train_X = model_representation(pret_model, train_corpus, clip_size)
print(train_X.size(), len(train_Y))


print(f" Train Representation Done")
val_X = model_representation(pret_model, val_corpus, clip_size)
print(f"Validation Representation Done")
test_X = model_representation(pret_model, test_corpus, clip_size)
print(f"Test Representation : Done")

print("Dataset Ready!")

###############################################################################
# Build the model
###############################################################################

#Setting model
if args.activation == 'None':
    activation = None
input_dim = len(train_X[0])
hidden_size = len(train_X)
print(f'input dim : {input_dim}')
model = LinearProbingModel(input_dim, hidden_size, (num_classe), activation)

#Setting optimizer and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.lr_patience)

###############################################################################
# Training code
###############################################################################

train_loss = train(model, criterion,train_X, train_Y[:2080])
data_path = os.path.join(project_base_path, "corpora","probing_pickled_files","traindata-"+args.pretrain_data+"-"+args.probe_task+".csv")
with open(data_path, 'w', newline='') as f:
    writer = csv.writer(f)
    for data in zip(train_X, train_Y[:2080]):
        data_x = np.append(data[0].detach().numpy().copy(),data[1].detach().numpy().copy())
        writer.writerow(data_x)
print(f'train data loss : {train_loss}')

#test_predicition = model(test_X)
#test_loss = criterion(test_predicition, test_Y)
#print(f"test data loss : {test_loss}")