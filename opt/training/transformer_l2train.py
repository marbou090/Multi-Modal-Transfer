import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
import os
import hashlib
import pickle
import json
from tqdm import tqdm

from corpora import data
from training import Transformer
from training.splitcross import SplitCrossEntropyLoss

from training.utils import batchify, repackage_hidden, get_slice

from paths import project_base_path

args = torch.load(("./args")) #ここにargsのデータの居場所を置く
#args.log_interval = 50
#args.valid_interval = 50
args.log_interval = 200
args.valid_interval = 1000
print(f"args: {args}")

eval_batch_size = 10
test_batch_size = 1

best_loss = 100000000
best_model = None
last_train_loss = -1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def l2_train(data, pret_model, pret_criterion, l1_test, seed,save_dir, run_name,l1_ntokens,l2_ntokens,
                  freeze_net=False, start_lr=30, check_epoch=5, lr_patience=5,
                  max_lr_decreases=1, cull_vocab=False, corpus_change="nothing"):

    save_path = os.path.join(save_dir, f"{run_name}-finetune.pickle")
    print(f"Finetuned model will save to {save_path}")

    global model, criterion, optimizer, scheduler, params
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    train_data, val_data, test_data = data
    model = pret_model.cuda()
    criterion = pret_criterion.cuda()
    params = list(model.parameters()) + list(criterion.parameters())
    optimizer = torch.optim.SGD(params, lr=start_lr, weight_decay=args.wdecay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=lr_patience, cooldown=1)
    if freeze_net:
        print("Freezing the neural network, just training embeddings")
        for name,param in model.named_parameters():
            if not ((name=='decoder.bias') or (name=='decoder.weight')):
                param.requires_grad = False
            else:
                print("-"*20)
                print(f"name: {name}")
                print(f"requires grad:{param.requires_grad} ")
            
    params = list(model.parameters()) + list(criterion.parameters())
    total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in params if x.size())
    print('Model total parameters:', total_params)

    
    epoch = 0
    num_lr_decreases = 0
    lr = start_lr
    val_loss_list = []
    overall_batch, epoch_batch, epoch_data_index = 0, 0, 0
    valid_time = time.time()
    stop_condition_met = False

    zero_shot_test = evaluate(test_data)

    ##########################################################################
    #fintuning
    ##########################################################################
    while not stop_condition_met:
        epoch_start_time = time.time()
        overall_batch, num_lr_decreases = \
            train(model, criterion, train_data, val_data, overall_batch,
                epoch_batch, epoch_data_index, valid_time, val_loss_list,
                scheduler, num_lr_decreases,l2_ntokens)
        epoch_batch, epoch_data_index = 0, 0
        epoch += 1
        #########################
        #num_lr_decreases=1
        #########################
        if num_lr_decreases >= max_lr_decreases:
            stop_condition_met = True
        if stop_condition_met == True:
            print("Stopping due to convergence")
        if epoch == check_epoch:
            print("Reached check epoch, calculating validation loss")
            train_loss_at_epoch = last_train_loss
            loss_at_epoch = evaluate(val_data)
            test_loss_at_epoch = evaluate(test_data)
            print(f"Loss {loss_at_epoch}, test loss {test_loss_at_epoch}")

    embeddings = model.encoder.weight.detach().cpu().numpy()

    print(f"Saving model to {save_path} ")
    with open(save_path, 'wb') as f:
        pickle.dump({'model':model, 
                    'criterion':criterion,
                     'optimizer':optimizer, 
                     'scheduler':scheduler,
                    'epoch':epoch, 
                    'num_lr_decreases':num_lr_decreases, 
                    'lr':lr, 
                    'best_loss':best_loss, 
                    'val_loss_list':val_loss_list,
                     'overall_batch':overall_batch, 
                     'epoch_batch':epoch_batch, 
                     'epoch_data_index':epoch_data_index,
                     'loss_at_epoch':loss_at_epoch,
                     'test_loss_at_epoch':test_loss_at_epoch,
                     'train_loss_at_epoch':train_loss_at_epoch,
                     #'zero_shot_test':zero_shot_test,
                     'embeddings':embeddings,
                     'last_train_loss':last_train_loss},
                   f)
    l1_test_loss = evaluate(l1_test)
    test_loss = evaluate(test_data)
    
    #NOTE this assumes that weights are tied, which they have been for all the
    #     experiments since the awd-lm main forces it.
    
    return val_loss_list, test_loss, last_train_loss, overall_batch, epoch, \
           loss_at_epoch, test_loss_at_epoch, train_loss_at_epoch, \
           zero_shot_test, l1_test_loss, embeddings

def model_load(fn):
    with open(fn, 'rb') as f:
        data= pickle.load(f)
    return data

"""
def model_save(fn, epoch, num_lr_decreases,lr, best_loss, val_loss_list,
                     overall_batch, epoch_batch, epoch_data_index,loss_at_epoch,test_loss_at_epoch,train_loss_at_epoch,
                     zero_shot_test,embeddings):
    with open(fn, 'wb') as f:
        torch.save([model, criterion, optimizer, scheduler,
                    (epoch, num_lr_decreases, lr, best_loss, val_loss_list,
                     overall_batch, epoch_batch, epoch_data_index,loss_at_epoch,test_loss_at_epoch,train_loss_at_epoch,
                     zero_shot_test,embeddings)],
                   f)
                   """
###############################################################################
# Training code
###############################################################################
def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            output = model(data)
            output = output.view(-1, args.num_embs)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(data_source) - 1)


criterion = nn.CrossEntropyLoss()

def train(model, criterion, train_data, val_data, overall_batch, epoch_batch,
          epoch_data_index, valid_time, val_loss_list, scheduler, num_lr_decreases,l2_ntokens):
    # Turn on training mode which enables dropout.
    global best_loss, best_model, last_train_loss

    model.train()
    total_loss = 0
    start_time = time.time()
    ntokens = l2_ntokens
    while epoch_data_index < train_data.size(0) - 1 - 1:
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        # Prevent excessively small or negative sequence lengths
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        # There's a very small chance that it could select a very long sequence length resulting in OOM
        seq_len = min(seq_len, args.bptt + 10)

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        model.train()
        data, targets = get_batch(train_data, epoch_data_index)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        
        optimizer.zero_grad()
        output= model(data)
        output = output.view(-1, args.num_embs)
        loss = criterion(output, targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args.clip: torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        total_loss += loss.item()
        optimizer.param_groups[0]['lr'] = lr2
        if overall_batch % args.log_interval == 0 and overall_batch > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                    'tr loss {:5.2f} | tr ppl {:8.2f} | bpc {:8.3f}'.format(
                epoch_batch, len(train_data) // args.bptt, optimizer.param_groups[-1]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss), cur_loss / math.log(2)))
            last_train_loss = cur_loss
            total_loss = 0
            start_time = time.time()

        if overall_batch % args.valid_interval == 0 and overall_batch > 0:
            elapsed = time.time() - valid_time
            val_loss = evaluate(val_data)
            val_loss_list.append(val_loss)
            scheduler.step(val_loss)
            if scheduler.in_cooldown:
                num_lr_decreases += 1
                print(f"Just decreased learning rate! Have decreased {num_lr_decreases} times.")
                print("Loading best model so far")
                model.load_state_dict(best_model)
            print('-' * 89)
            print('| validating at batch {:3d} | time: {:5.2f}m | valid loss {:5.2f} | '
                'valid ppl {:8.2f} | valid bpc {:8.3f}'.format(
              overall_batch, elapsed / 60, val_loss, math.exp(val_loss), val_loss / math.log(2)))
            print('-' * 89)
            if val_loss < best_loss:
                best_loss = val_loss
                print(f"New best loss {best_loss}")
                best_model = model.state_dict()
            ######################################
            #num_lr_decreases=1
            ############################################
            valid_time = time.time()
        ###
        epoch_batch += 1
        overall_batch += 1
        epoch_data_index += seq_len
    return overall_batch, num_lr_decreases