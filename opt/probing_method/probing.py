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

import training.model
from training.splitcross import SplitCrossEntropyLoss

from probing_method.utils import batchify, get_batch

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

def probing_train(data, pret_model,pret_criterion, seed, save_dir,run_name,start_lr=30, check_epoch=5, lr_patience=5,max_lr_decreases=1,):
    save_path = os.path.join(save_dir, f"{run_name}-probing.pickle")
    print(f"Probing model will save to {save_path}")

    global model, criterion, optimizer, scheduler, params
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    train_data, val_data, test_data = data
    model = pret_model.cuda()
    ##############################
    model.decoder = torch.nn.Linear(400,3 ).to('cuda:0')
    #############################
    criterion = pret_criterion.cuda()
    params = list(model.parameters()) + list(criterion.parameters())
    optimizer = torch.optim.AdamW(params)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience = lr_patience, cooldown=1)
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

    while not stop_condition_met:
        epoch_start_time = time.time()
        overall_batch, num_lr_decreases = \
            train(model, criterion, train_data, val_data, overall_batch,
                epoch_batch, epoch_data_index, valid_time, val_loss_list,
                scheduler, num_lr_decreases)
        epoch_batch, epoch_data_index = 0, 0
        epoch += 1

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
            ######################################
            break
            ############################################

    embeddings = model.encoder.weight.detach().cpu().numpy()

    test_loss = evaluate(test_data)

    return test_loss

###############################################################################
# Training code
###############################################################################

def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            targets = torch.tensor([1,0,0]).to('cuda:0')
            data = data.to('cuda:0')
            output = model(data)
            output = output.view(-1, 3).T
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(data_source) - 1)


def train(model, criterion, train_data, val_data, overall_batch, epoch_batch,
          epoch_data_index, valid_time, val_loss_list, scheduler, num_lr_decreases):
    global best_loss, best_model, last_train_loss

    model.train()
    total_loss = 0
    start_time = time.time()
    while epoch_data_index < train_data.size(0) - 1 - 1:
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        # Prevent excessively small or negative sequence lengths
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        # There's a very small chance that it could select a very long sequence length resulting in OOM
        seq_len = min(seq_len, args.bptt + 10)

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        model.train()
        data, _ = get_batch(train_data, epoch_data_index)
        targets = torch.tensor([1,0,0]).to('cuda:0')

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        optimizer.zero_grad()

        data = data.to('cuda:0')
        output= model(data)
        """
        y_pred = torch.max(output.view(-1, 3), dim=1)
        correct = 0
        for i in y_pred.indices:
            if i == 0:
                correct+=1
        print(f"correct num :{correct}\n case size:{len(y_pred.indices)}\n ratio:{correct/len(y_pred.indices)}")
        """
        loss = criterion(output.view(-1, 3).T, targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        if args.clip: torch.nn.utils.clip_grad_norm_(params, args.clip)
        optimizer.step()

        total_loss += loss.data
        optimizer.param_groups[0]['lr'] = lr2
        if overall_batch % args.log_interval == 0 and overall_batch > 0:
            cur_loss = total_loss.item() / args.log_interval
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
            val_loss = evaluate( val_data)
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

