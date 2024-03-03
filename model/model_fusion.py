import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb
import numpy as np
import os
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader
from druggraph import smiles2graph


class SimpleGNN(nn.Module):
    def __init__(self, node_feature_dim, output_dim):
        super(SimpleGNN, self).__init__()
        self.conv1 = GCNConv(node_feature_dim, 16)
        self.conv2 = GCNConv(16, output_dim)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

if __name__ == '__main__':
    smile = "NC1=NC(=O)N(C=C1)[C@@H]1O[C@H](CO)[C@@H](O)C1(F)F" 
    data = smiles2graph(smile)
    model = SimpleGNN(node_feature_dim=144, output_dim=2) # 确保node_feature_dim与实际数据一致
    loader = DataLoader([data], batch_size=1, follow_batch=['x', 'edge_index']) # follow_batch 不是必须的，只是一个例子
    for batch in loader:
        out = model(batch)
        print(out)
# class DrugPathFusion(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):


