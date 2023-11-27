import time
import torch
import numpy as np
from probing_method.utils import batchify, get_batch
from tqdm import tqdm

def probe_case(probing_model, embs_train, train_data, embs_test,
                           test_data):
    
    
    #define const
    embedding_size = len(embs_train)
    nr_of_categories = 3 #number of class
    device = 'cpu'

    #Define Probing Classifier
    probe = probing_model(embedding_size, nr_of_categories)
    
    
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(probe.parameters(), lr=30)
    metric = binary_acc
    epochs = 1

    for epoch in tqdm(range(epochs)):
        probe.train()
        embeddings = torch.from_numpy(embs_train).clone()
        labels = torch.tensor([1,0,0]).to(device)
        optimizer.zero_grad()
        preds = probe(embeddings)
        loss = loss_fn(preds, labels)
        loss.backward()
        optimizer.step()
        print(loss)


    print("Evaluating on test loader")
    avg_loss, total, avg_metric, per_label_acc = evaluate(probe, loss_fn, test_data, nr_of_categories, metric, use_confusion_matrix=True)
    return model, (avg_loss, total, avg_metric, per_label_acc)


def evaluate(model, loss_fn, valid_dl, num_classes, metric=None, use_confusion_matrix=False):
    if num_classes < 2:
        num_classes += 1
    confusion_matrix = torch.zeros(num_classes, num_classes).int()
    device = 'cpu'
    with torch.no_grad():
        model.eval()
        y_test = []
        results = []
        outputs = []
        for xb,yb in valid_dl:
            xb = xb.to(device)
            yb = yb.to(device)
            result = loss_batch(model, loss_fn, xb, yb, metric=metric)
            if use_confusion_matrix:
                _, preds = torch.max(result[0], 1)
                for t, p in zip(yb.view(-1), preds.view(-1)):
                        confusion_matrix[t.long(), p.long()] += 1
            results.append(result)
        s = confusion_matrix.sum(1)
        for i in range(len(s)):
            if not (s[i] > 0):
                s[i] == 1.0 #diag item will be 0 anyway
        s = s.float()
        per_label_acc = confusion_matrix.diag().float()/s
        data_size = confusion_matrix.sum()
        if use_confusion_matrix:
            print(confusion_matrix)
            print("Per-label accuracy: ",per_label_acc)
            print("Total accuracy: ",(confusion_matrix.diag().sum().float()/data_size).item())
        preds, losses, nums, metrics = zip(*results)

        total = np.sum(nums)
        avg_loss = np.sum(np.multiply(losses, nums))/total
        avg_metric = None
        if metric is not None:
            avg_metric = np.sum(np.multiply(metrics,nums))/total
    return avg_loss, total, avg_metric, per_label_acc

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_tag, dim = 1)
    correct_results_sum = (y_pred_tags == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    return acc.item()