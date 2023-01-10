import os
import warnings
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.nn import GCNConv, SGConv, SAGEConv, GATConv, GraphConv, GINConv
from torch_geometric.utils import degree, to_networkx
from torch_scatter import scatter

import networkx as nx


def get_base_model(name: str):
    def gat_wrapper(in_channels, out_channels):
        return GATConv(
            in_channels=in_channels,
            out_channels=out_channels // 4,
            heads=4
        )

    def gin_wrapper(in_channels, out_channels):
        mlp = nn.Sequential(
            nn.Linear(in_channels, 2 * out_channels),
            nn.ELU(),
            nn.Linear(2 * out_channels, out_channels)
        )
        return GINConv(mlp)

    base_models = {
        'GCNConv': GCNConv,
        'SGConv': SGConv,
        'SAGEConv': SAGEConv,
        'GATConv': gat_wrapper,
        'GraphConv': GraphConv,
        'GINConv': gin_wrapper
    }

    return base_models[name]


def get_activation(name: str):
    activations = {
        'relu': F.relu,
        'hardtanh': F.hardtanh,
        'elu': F.elu,
        'leakyrelu': F.leaky_relu,
        'prelu': torch.nn.PReLU(),
        'rrelu': F.rrelu
    }

    return activations[name]


def generate_split(num_samples: int, train_ratio: float, val_ratio: float, generator: torch.Generator = None):
    train_len = int(num_samples * train_ratio)
    val_len = int(num_samples * val_ratio)
    test_len = num_samples - train_len - val_len

    train_set, test_set, val_set = random_split(torch.arange(0, num_samples), (train_len, test_len, val_len),
                                                generator=generator)

    idx_train, idx_test, idx_val = train_set.indices, test_set.indices, val_set.indices
    train_mask = torch.zeros((num_samples,)).to(torch.bool)
    test_mask = torch.zeros((num_samples,)).to(torch.bool)
    val_mask = torch.zeros((num_samples,)).to(torch.bool)
    
    train_mask[idx_train] = True
    test_mask[idx_test] = True
    val_mask[idx_val] = True

    return train_mask, test_mask, val_mask


def save_model(model, optimizer, epoch, prefix, model_path):
    state_dict = OrderedDict({k: v for k, v in model.state_dict().items()})
    obj = {
        'epoch': epoch,
        'state_dict': state_dict,
        'optimizer': optimizer.state_dict()
    }
    model_name = os.path.join(model_path, f'{prefix}_ep{epoch}.pth')
    torch.save(obj, model_name)
    return model_name


def load_model(model, optimizer, model_name, device):
    cp = torch.load(model_name, map_location=device)
    model.load_state_dict(
        OrderedDict({k: v for k, v in cp['state_dict'].items()}), strict=False)
    optimizer.load_state_dict(cp['optimizer'])
    return model, optimizer, cp['epoch']


def remove_model(epoch, prefix, model_path):
    model_name = f'{prefix}_ep{epoch}.pth'
    try:
        os.remove(os.path.join(model_path, model_name))
    except FileNotFoundError:
        warnings.warn('No model file is removed.')
