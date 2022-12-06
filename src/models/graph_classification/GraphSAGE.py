import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from readouts.basic_readout import readout_function

"""
Base paper: https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf
"""

class GraphSAGE(nn.Module):

    def __init__(self, n_feat, n_class, n_layer, agg_hidden, fc_hidden, dropout, readout, device,
    aggregation='mean'):
        
        super(GraphSAGE, self).__init__()
        
        self.n_layer = n_layer
        self.dropout = dropout
        self.readout = readout
        self.aggregation = aggregation
        self.device = device
        self.readout_dim = agg_hidden * n_layer
        
        # Graph sage layer
        self.graph_sage_layers = []
        for i in range(n_layer):
            if i == 0:
                sage = SAGEConv(n_feat, agg_hidden).to(device)
            else:
                sage = SAGEConv(agg_hidden, agg_hidden).to(device)
            sage.aggr = self.aggregation
            self.graph_sage_layers.append(sage)
        
        if self.aggregation == 'max':
            self.fc_max = nn.Linear(agg_hidden, agg_hidden)
        
        # Fully-connected layer
        self.fc1 = nn.Linear(self.readout_dim, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, n_class)
    
    def preprocessing(self, edge_matrix_list, x_list, node_count_list, edge_matrix_count_list):
        total_edge_matrix_list = []
        start_edge_matrix_list = []
        end_edge_matrix_list = []
        batch_list = []
        total_x_list = []
        
        max_value = torch.tensor(0).to(self.device)
        for i, edge_matrix in enumerate(edge_matrix_list):
            for a in range(edge_matrix_count_list[i]):
                start_edge_matrix_list.append(max_value + edge_matrix[a][0])
                end_edge_matrix_list.append(max_value + edge_matrix[a][1])
            if max_value < max_value + edge_matrix[edge_matrix_count_list[i] - 1][0]:
                max_value = max_value + edge_matrix[edge_matrix_count_list[i] - 1][0]
        total_edge_matrix_list.append(start_edge_matrix_list)
        total_edge_matrix_list.append(end_edge_matrix_list)
        
        for i in range(len(x_list)):
            for a in range(node_count_list[i]):
                batch_list.append(i)
                total_x_list.append(x_list[i][a].cpu().numpy())
        
        return torch.tensor(total_edge_matrix_list).long().to(self.device), torch.tensor(batch_list).float().to(self.device), torch.tensor(total_x_list).float().to(self.device)
              
    def forward(self, data):
        x, adj = data[:2]
        edge_matrix_list = data[6]
        node_count_list = data[7]
        edge_matrix_count_list = data[8]
        total_edge_matrix_list, batch_list, total_x_list = self.preprocessing(edge_matrix_list, x, node_count_list, edge_matrix_count_list)
        
        x_list = []
        x = total_x_list
        
        for i in range(self.n_layer):
           
           # Graph sage layer
           x = F.relu(self.graph_sage_layers[i](x, total_edge_matrix_list))
           if self.aggregation == 'max':
               x = torch.relu(self.fc_max(x))
           
           # Dropout
           if i != self.n_layer - 1:
               x = F.dropout(x, p=self.dropout, training=self.training)
             
           x_list.append(x)
        
        x = torch.cat(x_list, dim=1)
           
        # Readout
        x = readout_function(x, self.readout, batch=batch_list, device=self.device)
        x = x.reshape(adj.size()[0], self.readout_dim)
        
        # Fully-connected layer
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x))

        return x

    def __repr__(self):
        layers = ''
        
        for i in range(self.n_layer):
            layers += str(self.graph_sage_layers[i]) + '\n'
        layers += str(self.fc1) + '\n'
        layers += str(self.fc2) + '\n'
        return layers