from __future__ import print_function
from __future__ import division
import numpy as np
import torch
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F
import time
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm.notebook import tnrange
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from scipy.sparse import diags
from scipy.sparse import eye
from pathlib import Path
from functools import partial

import sys
import json
import os
import src.models.graph_classification.test_model as test_model

def main(targets):
    if 'test' in targets:
        with open('config/test_param.json') as fh:
            test_params = json.load(fh)

        # run test on cora dataset using GCN
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        path = Path(test_params.get('data_path'))
        paper_features_label = np.genfromtxt(path/'cora.content', dtype=np.str)

        features = csr_matrix(paper_features_label[:, 1:-1], dtype=np.float32)
        labels = paper_features_label[:, -1]
        lbl2idx = {k:v for v,k in enumerate(sorted(np.unique(labels)))}
        labels = [lbl2idx[e] for e in labels]

        papers = paper_features_label[:,0].astype(np.int32)

        paper2idx = {k:v for v,k in enumerate(papers)}
        edges = np.genfromtxt(path/'cora.cites', dtype=np.int32)
        edges = np.asarray([paper2idx[e] for e in edges.flatten()], np.int32).reshape(edges.shape)

        adj = coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                 shape=(len(labels), len(labels)), dtype=np.float32)

        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = test_model.normalize(adj + eye(adj.shape[0]))

        adj = torch.FloatTensor(adj.todense())
        features = torch.FloatTensor(features.todense())
        labels = torch.LongTensor(labels)

        np.random.seed(42)
        n_train = 200
        n_val = 300
        n_test = len(features) - n_train - n_val
        idxs = np.random.permutation(len(features))
        idx_train = torch.LongTensor(idxs[:n_train])
        idx_val   = torch.LongTensor(idxs[n_train:n_train+n_val])
        idx_test  = torch.LongTensor(idxs[n_train+n_val:])
        
        adj = adj.to(device)
        features = features.to(device)
        labels = labels.to(device)
        idx_train = idx_train.to(device)
        idx_val = idx_val.to(device)
        idx_test = idx_test.to(device)

        n_labels = labels.max().item() + 1
        n_features = features.shape[1]

        torch.manual_seed(34)

        model = test_model.GCN(nfeat=n_features,
                    nhid=20, #hidden = 16
                    nclass=n_labels,
                    dropout=0.5) #dropout = 0.5

        model = model.to(device)
        optimizer = optim.Adam(model.parameters(),
                            lr=0.001, weight_decay=5e-4)


        def accuracy(output, labels):
            preds = output.max(1)[1].type_as(labels)
            correct = preds.eq(labels).double()
            correct = correct.sum()
            return correct / len(labels)

        def step():
            t = time.time()
            model.train()
            optimizer.zero_grad()
            output = model(features, adj)
            loss = F.nll_loss(output[idx_train], labels[idx_train])
            acc = accuracy(output[idx_train], labels[idx_train])
            loss.backward()
            optimizer.step()
            
            return loss.item(), acc

        def evaluate(idx):
            model.eval()
            output = model(features, adj)
            loss = F.nll_loss(output[idx], labels[idx])
            acc = accuracy(output[idx], labels[idx])
            
            return loss.item(), acc

        epochs = 1000
        print_steps = 100
        train_loss, train_acc = [], []
        val_loss, val_acc = [], []

        for i in tnrange(epochs):
            tl, ta = step()
            train_loss += [tl]
            train_acc += [ta]
            
            if((i+1)%print_steps) == 0 or i == 0:
                tl, ta = evaluate(idx_train)
                vl, va = evaluate(idx_val)
                val_loss += [vl]
                val_acc += [va]
                
                print('Epochs: {}, Train Loss: {:.3f}, Train Acc: {:.3f}, Validation Loss: {:.3f}, Validation Acc: {:.3f}'.format(i, tl, ta, vl, va))

        print('Test passed!')

if __name__ == '__main__':
    targets = sys.argv[1:]
    main(targets)