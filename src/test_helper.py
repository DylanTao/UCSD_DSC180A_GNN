import os
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

# Path: src/test_helper.py
def load_dataset(save_path, name, percent = 0.8):

    dataset = TUDataset(root=save_path, name=name)
    dataset = dataset.shuffle()
    train_dataset = dataset[:int(len(dataset) * percent)]
    test_dataset = dataset[int(len(dataset) * percent):]

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader