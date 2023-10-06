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

from training.utils import batchify, get_batch, repackage_hidden, get_slice

from paths import project_base_path

def probing_train(data, pret_model,pret_criterion, seed, save_dir,run_name):
    save_path = os.path.join(save_dir, f"{run_name}-probing.pickle")
    print(f"Probing model will save to {save_path}")
    