import argparse
import os
import numpy as np
import pickle
import torch

from paths import project_base_path
from training.utils import batchify, get_batch, repackage_hidden, get_slice
from training.probing import probing_train


parser = argparse.ArgumentParser()
parser.add_argument('--run-name', type=str, help='how to call tha save fill for this run')
parser.add_argument('--pretrain', type=str, help="Which pretrained model")
parser.add_argument('--trial', type=int, default=0,
                    help='trial  (of pretrained models)')
parser.add_argument('--probe-task', type=str, default='Case', help='which probe task')
parser.add_argument('--seed', type=int, default=4, help="Seed will be args.seed*100 + the pretrain_index")
args = parser.parse_args()
args.cuda = True
print(args)

possible_probetask = \
    ['Case','Definite','Degree','ExtPos','Gender','Mood','Number','NumForm','NumType','Person','PronType','Tense','VerbForm']
assert args.probe_task in possible_probetask

batch_size =80

def run():
    pretrain_path = os.path.join(project_base_path, "models", "pretrained_models", args.pretrain)
    pretrain_idx = possible_probetask.index(args.probe_task)
    save_dir = os.path.join(project_base_path, "models", "probes", args.pretrain)
    if not os.path.exists(save_dir):
            os.mkdir(save_dir)
    save_path = os.path.join(save_dir, args.run_name)
    print(f"Will save to {save_path}")

    probing_data_location = os.path.join(project_base_path, "corpora", "pickled_files", f"corpus-{args.probe_task}")
    corpus = load_corpus(probing_data_location)
    train_data = batchify(corpus.train, batch_size, args)
    val_data = batchify(corpus.valid, batch_size, args)
    test_data = batchify(corpus.test, batch_size, args)

    seed = args.seed*100 + pretrain_idx
    np.random.seed(seed)

    print(f"Starting {args.pretrain}-{args.trial} : Probe tasks {args.probe_task}")
    model_path = os.path.join(pretrain_path, f"trial{str(args.trial)}")
    with open(model_path, 'rb') as f:
            model, criterion, optimizer, scheduler, run_data = torch.load(f)
    test_loss = probing_train((train_data,val_data,test_data), model, criterion, seed, save_dir=save_dir, run_name=args.run_name)

    print(test_loss)




def load_corpus(data_path, cull_vocab=False, shuffle_vocab=False):
    if cull_vocab:
        data_path = data_path + ".cull"
    if shuffle_vocab:
        assert cull_vocab, "Usually don't have unculled shuffled corpora"
        data_path = data_path + "-shuf"

    corpus = torch.load(data_path)
    return corpus
        


if __name__ == "__main__":
        run()

