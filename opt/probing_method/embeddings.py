import numpy as np
from training.utils import repackage_hidden

def compute_embeddings(embedder, train_data, val_data,batch_size,
                        merge_funcs=[]):
    all_embs=[]
    print(train_data)
    print(train_data.shape)
    hidden = embedder.init_hidden(157)
    embs, _ = embedder(train_data,hidden)
    hidden = embedder.init_hidden(1)
    embs_test, _ = embedder(val_data,hidden)
    print(embs)
    print(embs_test)
    exit()

    embs = embs.detach().numpy()
    embs_test = embs_test.detach().numpy()
    all_embs.append((embs[0], embs_test[0]))
    if (len(embs) > 1):
        all_embs.append((embs[1],embs_test[1]))
        for f in merge_funcs:
            all_embs.append((f(embs),f(embs_test)))

    return all_embs

def average_embeddings(embs):
    return np.mean(embs,axis=0)

def concatenate_embeddings(embs):
    return np.concatenate(embs, axis=1)