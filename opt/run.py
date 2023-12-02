# This is the pretrain.py from the tilt-transfer repository
import argparse
import numpy as np
import os
import json
import collections
import time

import torch
from torch import nn

from paths import project_base_path
from ptobing_method.probing_model import MultiLaryerProbingModel,LinearProbingModel
from training.utils import repackage_hidden

parser = argparse.ArgumentParser()

parser.add_argument('--probe-task', type=str, default='BigramShift',
                    help='Short name of the probe task')
parser.add_argument('--probe-model', type=str, default='MLP',
                    help='type of probe model (MLP, Linear)')
parser.add_argument('--pretrain-data', type=str, default='wiki-es',
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

def probe_data_load(fn):    
    train_data = []
    val_data = []
    test_data = []
    with open(fn, 'r') as f:
        for line in f:
            sentences = line.split(' ')
            if sentences[0] == ("tr"):
                train_data.append(sentences[2])
            elif sentences[0] == ("va"):
                val_data.append(sentences[2])
            elif sentences[0] == ("te"):
                test_data.append(sentences[2])
    return train_data, val_data, test_data

def probe_label_load(fn):    
    train_data = []
    val_data = []
    test_data = []
    with open(fn, 'r') as f:
        for line in f:
            sentences = line.split(' ')
            if sentences[0] == ("tr"):
                train_data.append(sentences[1])
            elif sentences[0] == ("va"):
                val_data.append(sentences[1])
            elif sentences[0] == ("te"):
                test_data.append(sentences[1])
    return train_data, val_data, test_data

def model_representation(model, data, batch_size=80):
    representations = []
    for line in data:
        hidden = model.init_hidden(batch_size)
        hidden = repackage_hidden(hidden)
        representation, _, _, _ = model(line, hidden, return_h=True)
        representations.append(representation)
    return representations

#####Train
def train(model, criterion, data_X, data_Y):
    model.train()
    optimizer.zero_grad()
    predictions = model(data_X)
    loss = criterion(predictions, data_Y)
    index = index + 1
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
            start_time = time.time()
        
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
# Load data
###############################################################################


pretraining_path = os.path.join(project_base_path, "models", "l2_results", "RNN", args.pretrain_data,args.pretrain_model)
pret_model, _, _, _, run_data = model_load(pretraining_path)

probe_path = os.path.join(project_base_path, "corpora","pickled_files",args.probe_task)
train_data, val_data, test_data = probe_data_load(probe_path)
train_Y, val_Y, test_Y = probe_label_load(probe_path)
num_classes = collections.Counter(train_Y)

print("Generating Probe Task Data Set")
#学習済みに文章いれて内部表現を保存していく
train_X = model_representation(pret_model, train_data)
val_X = model_representation(pret_model, val_data)
test_X = model_representation(pret_model, test_data)


print("Dataset Ready!")
print(f"Train Original Text : {train_data[0]}\n Representation : {train_X[0]}\n Label : {train_Y[0]}")

###############################################################################
# Build the model
###############################################################################

#Setting model
if args.activation == 'None':
    activation = None
input_dim = len(train_X[0])
model = LinearProbingModel(input_dim, num_classes, activation)

#Setting optimizer and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=args.lr_patience)

###############################################################################
# Training code
###############################################################################

train_loss = train(model, criterion,train_X, train_Y)
print(f'train data loss : {train_loss}')

test_predicition = model(test_X)
test_loss = criterion(test_predicition, test_Y)
print(f"test data loss : {test_loss}")